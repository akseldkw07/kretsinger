1. Github

```bash
sudo apt update
sudo apt install gh
```

2. zsh

```bash
sudo apt install zsh

```

3. oh-my-zsh

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

```

4. Powerlevel10k

```bash
p10k configure
```

5. zsh plugins

```bash
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

source ~/.oh-my-zsh/custom/themes/powerlevel10k/powerlevel10k.zsh-theme && p10k configure
```

6. homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

7. micromamba

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

NOTE create env in separate window, is slow

8. nodejs
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
```

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
```

```bash
nvm install --lts
```

```bash
sudo apt-get install -y unzip

curl -fsSL https://bun.sh/install | bash
```

9. claude

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

10. ripgrep

```bash
sudo apt install ripgrep
```

11. git config setup

```bash
cp ~/coding/kretsinger/backup/.gitconfig ~/.gitconfig
```

12. Claude plugins

```bash

/plugin marketplace add thedotmack/claude-mem

/plugin install claude-mem
```
