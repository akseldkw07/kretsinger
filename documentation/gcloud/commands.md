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
