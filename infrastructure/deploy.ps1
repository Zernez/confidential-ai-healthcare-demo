Param(
  [string]$Account = "fernando.pannullo@collaboratore.uniparthenope.it",
  [string]$RegionPref = "",
  [string]$Rg = "rg-tdx-h100-dist",
  [string]$Vnet = "vnet-tdx-h100",
  [string]$SubnetCpu = "snet-cpu",
  [string]$SubnetGpu = "snet-gpu",
  [string]$NsgCpu = "nsg-cpu",
  [string]$NsgGpu = "nsg-gpu",
  [string]$VmCpu = "vm-tdx-cpu",
  [string]$VmGpu = "vm-h100-gpu",
  [string]$AdminUser = "azureuser",
  [string]$SizeCpu = "Standard_DCadsv6", #AMD SEV
  [string]$SizeCpu = "Standard_DC2es_v6", #INTEL TDX
  [string]$SizeGpu = "Standard_NCC40ads_H100_v5",
  [string]$ImageCPU = "Canonical:0001-com-ubuntu-confidential-vm-jammy:22_04-lts-cvm:latest",
  [string]$ImageGPU = "Canonical:0001-com-ubuntu-confidential-vm-jammy:22_04-lts-cvm:latest"
)

# Login e subscription
az login
$subId = az account list --query "[?user.name=='$Account'].id" -o tsv
if (-not $subId) { throw "Subscription per $Account non trovata" }
az account set --subscription $subId

# Regione
$preferredRegions = if ($RegionPref) { @($RegionPref) } else { @("westeurope","northeurope","uksouth","switzerlandnorth","germanywestcentral") }
$availableRegions = az account list-locations --query "[].name" -o tsv
$region = $preferredRegions | Where-Object { $availableRegions -contains $_ } | Select-Object -First 1
if (-not $region) { throw "Nessuna regione disponibile tra: $($preferredRegions -join ', ')" }
Write-Host "Regione selezionata: $region"

# Resource group
if (-not (az group show -n $Rg --query name -o tsv 2>$null)) {
  az group create -n $Rg -l $region | Out-Null
}

# VNet
if (-not (az network vnet show -g $Rg -n $Vnet --query name -o tsv 2>$null)) {
  az network vnet create -g $Rg -n $Vnet --address-prefixes 10.30.0.0/16 `
    --subnet-name $SubnetCpu --subnet-prefix 10.30.1.0/24 | Out-Null
}

# Subnet GPU
if (-not (az network vnet subnet show -g $Rg --vnet-name $Vnet -n $SubnetGpu --query name -o tsv 2>$null)) {
  az network vnet subnet create -g $Rg --vnet-name $Vnet -n $SubnetGpu --address-prefixes 10.30.2.0/24 | Out-Null
}

# NSG
foreach ($nsg in @($NsgCpu, $NsgGpu)) {
  if (-not (az network nsg show -g $Rg -n $nsg --query name -o tsv 2>$null)) {
    az network nsg create -g $Rg -n $nsg -l $region | Out-Null
  }
}

# Regole NSG
$myIp = (Invoke-RestMethod -Uri "https://api.ipify.org").ToString()
az network nsg rule create -g $Rg --nsg-name $NsgCpu -n "Allow-SSH" --priority 1000 `
  --access Allow --protocol Tcp --direction Inbound --source-address-prefixes $myIp `
  --source-port-ranges "*" --destination-address-prefixes "*" --destination-port-ranges 22 2>$null | Out-Null

az network nsg rule create -g $Rg --nsg-name $NsgGpu -n "Allow-SSH" --priority 1000 `
  --access Allow --protocol Tcp --direction Inbound --source-address-prefixes $myIp `
  --source-port-ranges "*" --destination-address-prefixes "*" --destination-port-ranges 22 2>$null | Out-Null

az network nsg rule create -g $Rg --nsg-name $NsgGpu -n "Allow-CPU-HTTPS" --priority 1100 `
  --access Allow --protocol Tcp --direction Inbound --source-address-prefixes 10.30.1.0/24 `
  --source-port-ranges "*" --destination-address-prefixes "*" --destination-port-ranges 8443 2>$null | Out-Null

# Associa NSG
az network vnet subnet update -g $Rg --vnet-name $Vnet -n $SubnetCpu --network-security-group $NsgCpu | Out-Null
az network vnet subnet update -g $Rg --vnet-name $Vnet -n $SubnetGpu --network-security-group $NsgGpu | Out-Null

# Public IPs
foreach ($vm in @($VmCpu, $VmGpu)) {
  $pipName = "pip-$vm"
  if (-not (az network public-ip show -g $Rg -n $pipName --query name -o tsv 2>$null)) {
    az network public-ip create -g $Rg -n $pipName --sku Standard --allocation-method Static | Out-Null
  }
}

# NICs
foreach ($vm in @($VmCpu, $VmGpu)) {
  $nicName = "nic-$vm"
  $subnetName = if ($vm -eq $VmCpu) { $SubnetCpu } else { $SubnetGpu }
  $pipName = "pip-$vm"
  if (-not (az network nic show -g $Rg -n $nicName --query name -o tsv 2>$null)) {
    az network nic create -g $Rg -n $nicName --vnet-name $Vnet --subnet $subnetName --public-ip-address $pipName | Out-Null
  }
}

$nicCpuId = az network nic show -g $Rg -n "nic-$VmCpu" --query id -o tsv
$nicGpuId = az network nic show -g $Rg -n "nic-$VmGpu" --query id -o tsv

# VM GPU (Trusted Launch, Secure Boot ON)
if (-not (az vm show -g $Rg -n $VmGpu --query name -o tsv 2>$null)) {
  az vm create -g $Rg -n $VmGpu `
    --image $ImageGPU `
    --size $SizeGpu `
    --nics $nicGpuId `
    --admin-username $AdminUser `
    --generate-ssh-keys `
    --security-type ConfidentialVM `
    --enable-vtpm true `
    --enable-secure-boot true `
    --os-disk-security-encryption-type DiskWithVMGuestState | Out-Null
}

# --- PHASE 1: install NVIDIA driver via ubuntu-drivers (signed; Secure Boot-friendly) ---
$installDriver = @(
  "set -e",
  "export DEBIAN_FRONTEND=noninteractive",
  "apt-get update",
  "apt-get install -y ubuntu-drivers-common",
  "ubuntu-drivers install",
  "echo 'Driver install triggered; reboot required...'"
)

Start-Sleep -Seconds 20
# az vm run-command invoke -g $Rg -n $VmGpu --command-id RunShellScript --scripts $installDriver | Out-Null

# Reboot to load signed NVIDIA kernel modules
#az vm restart -g $Rg -n $VmGpu | Out-Null
# Start-Sleep -Seconds 20

# --- PHASE 2: Docker + NVIDIA Container Toolkit + repo clone ---
$installContainer = @(
  "set -e",
  "export DEBIAN_FRONTEND=noninteractive",
  "apt-get update",
  # Install prerequisites
  "apt-get install -y ca-certificates curl gnupg lsb-release git",
  # Add Docker's official GPG key
  "install -m 0755 -d /etc/apt/keyrings",
  "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
  "chmod a+r /etc/apt/keyrings/docker.gpg",
  # Set up Docker repository
  "echo ""deb [arch=`$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu `$(lsb_release -cs) stable"" > /etc/apt/sources.list.d/docker.list",
  
  
  # Install Docker Engine
  # "apt-get update",
  # "apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
  # NVIDIA Container Toolkit repo keyring and list
  # "install -d /usr/share/keyrings",
  # "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
  # "curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/libnvidia-container.list " +
  #   "| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' " +
  #   "> /etc/apt/sources.list.d/nvidia-container-toolkit.list",
  # "apt-get update",
  # "apt-get install -y nvidia-container-toolkit",
  # Configure Docker runtime
  # "nvidia-ctk runtime configure --runtime=docker || true",
  "systemctl restart docker",
  # Clone repo
  "cd /home/${AdminUser}",
  "if [ ! -d project ]; then git clone https://github.com/Zernez/confidential-ai-healthcare-demo.git project; fi",
  "chown -R ${AdminUser}:${AdminUser} /home/${AdminUser}/project"
  # Basic host verification
  # "nvidia-smi || true"
)

az vm run-command invoke -g $Rg -n $VmGpu --command-id RunShellScript --scripts $installContainer | Out-Null

Write-Host "Provisioning completato. Esegui 'nvidia-smi' via SSH sulla VM per verificare i driver e prova un container: docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi"