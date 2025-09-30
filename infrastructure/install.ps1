Param(
  [string]$Rg = "rg-tdx-h100-dist",
  [string]$VmGpu = "vm-h100-gpu",
  [string]$AdminUser = "azureuser"
)

# Script Bash da eseguire nella VM
$installScriptLines = @(
  "set -e",
  "export DEBIAN_FRONTEND=noninteractive",
  "apt-get update",
  "apt-get install -y build-essential curl jq ca-certificates gnupg lsb-release pciutils net-tools unzip python3 python3-pip git",
  'distribution=$(. /etc/os-release; echo $ID$VERSION_ID)',
  "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
  "dpkg -i cuda-keyring_1.1-1_all.deb || true",
  "apt-get update",
  "apt-get -y install cuda-toolkit-12-2",
  "apt-get -y install nvidia-driver-535 || true",
  "apt-get -y install docker.io",
  "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
  "curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list",
  "nvidia-ctk runtime configure --runtime=docker || true",
  "systemctl restart docker",
  # --- CLONE REPO ---
  "cd /home/$AdminUser",
  "if [ ! -d project ]; then git clone https://github.com/Zernez/confidential-ai-healthcare-demo.git project; fi",
  'chown -R $AdminUser:$AdminUser project',
  'rm -f cuda-keyring_1.1-1_all.deb',
  'apt-get clean'
)

# Esecuzione remota sulla VM GPU
Write-Host "Esecuzione installazione software su VM: $VmGpu nel gruppo: $Rg"
az vm run-command invoke `
  --resource-group $Rg `
  --name $VmGpu `
  --command-id RunShellScript `
  --scripts $installScriptLines
