## Commands

### gcloud

`Sign-in`

gcloud compute ssh --zone "us-east1-d" "l4-default" --tunnel-through-iap --project "adl-6691"

`Start`

gcloud compute instances start l4-default \
  --zone us-east1-d \
  --project adl-6691

`Stop`

gcloud compute instances stop l4-default --zone us-east1-d --project adl-6691

`setup`

$(gcloud info --format="value(basic.python_location)") -m pip install numpy

$


**Allow nvidia driver downloads**

gcloud compute networks subnets update default \
    --region=us-east1 \
    --enable-private-ip-google-access

gcloud compute project-info add-metadata \
  --metadata serial-port-enable=TRUE \
  --project adl-6691

**Cloud Router & NAT Gateway**

gcloud compute routers create router-us-east1 \
    --network=default \
    --region=us-east1

gcloud compute routers nats create nat-gateway-us-east1 \
    --router=router-us-east1 \
    --region=us-east1 \
    --auto-allocate-nat-external-ip-addresses \
    --nat-all-subnet-ip-ranges


**Brew, micrombamba**
```bash
sudo apt-get update
sudo apt-get install build-essential procps curl file git -y


/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

(echo; echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"') >> /home/$USER/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

./bin/micromamba shell init --shell bash --root-prefix ~/micromamba
source ~/.bashrc

```

### l4-32vcpu-16core-128ram-250disk (columbia-492622)

`Start`

```bash
gcloud compute instances start l4-32vcpu-16core-128ram-250disk \
  --zone us-central1-a \
  --project columbia-492622
```

`SSH`

```bash
gcloud compute ssh --zone "us-central1-a" "l4-32vcpu-16core-128ram-250disk" --tunnel-through-iap --project "columbia-492622"
```

### Create l4-32vcpu-16core-128ram-250disk

```bash

gcloud compute instances create l4-32vcpu-16core-128ram-250disk --project=columbia-492622 --zone=us-central1-a --machine-type=g2-standard-32 --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default --can-ip-forward --maintenance-policy=TERMINATE --provisioning-model=STANDARD --instance-termination-action=STOP --max-run-duration=28800s --service-account=117142832219-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append --accelerator=count=1,type=nvidia-l4 --tags=http-server,https-server --create-disk=auto-delete=yes,boot=yes,device-name=l4-32vcpu-16core-128ram-250disk,image=projects/ml-images/global/images/common-cu128-ubuntu-2404-nvidia-570-v20260323,mode=rw,size=250,type=pd-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --labels=goog-ec-src=vm_add-gcloud --reservation-affinity=any && gcloud compute resource-policies create snapshot-schedule default-schedule-1 --project=columbia-492622 --region=us-central1 --max-retention-days=14 --on-source-disk-delete=keep-auto-snapshots --daily-schedule --start-time=12:00 && gcloud compute disks add-resource-policies l4-32vcpu-16core-128ram-250disk --project=columbia-492622 --zone=us-central1-a --resource-policies=projects/columbia-492622/regions/us-central1/resourcePolicies/default-schedule-1
```
