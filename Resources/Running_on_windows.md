# Running on windows

To run the algorithm on a windows computer you need to use a windows subsytem please follow the instructions at the following link: https://docs.microsoft.com/en-us/windows/wsl/install-win10. 

We have tested [Ubuntu 18.04 LTS](https://www.microsoft.com/en-gb/p/ubuntu-1804-lts/9n9tngvndl3q?rtc=1#activetab=pivot:overviewtab) and shown it works so it is prefered over other distributions.

# Setup subsytem
Once your windows subsytem is installed and you're logged-in and inside your ~ (home) directory run the following:
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
To download the latest version of miniconda.

```
bash ./Miniconda3-latest-Linux-x86_64.sh
```
To install miniconda (please select default options).

```
export PATH=~/miniconda3/bin:$PATH
```
To activate conda comand.

# Install AutoDot
## From github (only available when repo is public)
```
git clone https://github.com/oxquantum-repo/AutoDot.git
```
This will download a copy of the current repo. After this subsytem setup is complete.

## From zip
