Param(
  [string]$Account = "fernando.pannullo@hotmail.com",
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
  [string]$SizeCpu = "Standard_DC2es_v6",
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

# VM GPU
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


# Installazione pacchetti + clone repo
$installScriptLines = @(
  "set -e",
  "apt-get update",
  "apt-get install -y build-essential curl jq ca-certificates gnupg lsb-release pciutils net-tools unzip python3 python3-pip git",
  'distribution=$(. /etc/os-release; echo $ID$VERSION_ID)',
#  'wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb',
  'wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb',
  "dpkg -i cuda-keyring_1.1-1_all.deb || true",
  "apt-get update",
  "apt-get -y install cuda-toolkit-12-2",
  "apt-get -y install nvidia-driver-535 || true",
  "apt-get -y install tdx-qgs-vm libtdx-attest || true",
  "apt-get -y install docker.io",
  'distribution=$(. /etc/os-release; echo $ID$VERSION_ID)',
  "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
  'curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed \"s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g\" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list',
  "apt-get update && apt-get install -y nvidia-container-toolkit || true",
  "nvidia-ctk runtime configure --runtime=docker || true",
  "systemctl restart docker",
  # --- CLONE DELLA REPO ---
  "cd /home/azureuser",
  "git clone https://github.com/Zernez/confidential-ai-healthcare-demo.git project",
  "chown -R azureuser:azureuser project"
)

# Wait for VMs to be fully provisioned
Start-Sleep -Seconds 30

az vm run-command invoke -g $Rg -n $VmGpu --command-id RunShellScript --scripts $installScriptLines
