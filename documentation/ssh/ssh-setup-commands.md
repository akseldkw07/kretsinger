1. Github

```bash
sudo apt update
sudo apt install gh
```

2. micromamba

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

NOTE create env in separate window, is slow

3. claude

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

4. homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

4. zsh

```bash
sudo apt install zsh

```

5. oh-my-zsh

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

```

6. Powerlevel10k

```bash
p10k configure
```

7. zsh plugins

```bash
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

source ~/.oh-my-zsh/custom/themes/powerlevel10k/powerlevel10k.zsh-theme && p10k configure
```

8. ripgrep

```bash
sudo apt install ripgrep
```

9. git config setup

```bash
cp ~/coding/kretsinger/backup/.gitconfig ~/.gitconfig
```
