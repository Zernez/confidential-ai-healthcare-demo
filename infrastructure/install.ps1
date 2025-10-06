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
  # Install prerequisites
  "apt-get install -y ca-certificates curl gnupg lsb-release git",
  # Rimuovi vecchie versioni di Docker se presenti
  "apt-get remove -y docker docker-engine docker.io containerd runc || true",
  # Add Docker's official GPG key
  "install -m 0755 -d /etc/apt/keyrings",
  "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
  "chmod a+r /etc/apt/keyrings/docker.gpg",
  # Set up Docker repository - versione statica per Ubuntu 22.04 jammy
  "echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable' > /etc/apt/sources.list.d/docker.list",
  # Installa Docker Engine e plugin
  "apt-get update",
  "apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
  # Abilita e avvia Docker
  "systemctl enable docker",
  "systemctl start docker",
  # Installa NVIDIA Container Toolkit e configura Docker per GPU
  "install -d /usr/share/keyrings",
  "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
  "curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list",
  "apt-get update",
  "apt-get install -y nvidia-container-toolkit",
  "nvidia-ctk runtime configure --runtime=docker || true",
  "systemctl restart docker",
  # --- CLONE REPO ---
  "cd /home/$AdminUser",
  "if [ ! -d project ]; then git clone https://github.com/Zernez/confidential-ai-healthcare-demo.git project; fi",
  'chown -R $AdminUser:$AdminUser project',
  'apt-get clean'
)

# Esecuzione remota sulla VM GPU
Write-Host "Esecuzione installazione software su VM: $VmGpu nel gruppo: $Rg"
az vm run-command invoke `
  --resource-group $Rg `
  --name $VmGpu `
  --command-id RunShellScript `
  --scripts $installScriptLines
