# This is a test

# Installing Docker on Raspberry Pi 4 (8GB RAM) with Raspberry Pi OS 64-bit

Follow these steps to install Docker on your Raspberry Pi 4:

## Prerequisites

- Raspberry Pi 4 with 8GB RAM.
- Raspberry Pi OS 64-bit installed and updated.
- Internet connection.
- Terminal access.

## Steps

1. **Update the System**

    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2. **Install Required Packages**

    ```bash
    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
    ```

3. **Add Docker's Official GPG Key**

    ```bash
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    ```

4. **Set Up the Docker Repository**

    ```bash
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian bullseye stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```

5. **Install Docker**

    ```bash
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io
    ```

6. **Verify Docker Installation**

    ```bash
    sudo docker --version
    ```

7. **Enable and Start Docker Service**

    ```bash
    sudo systemctl enable docker
    sudo systemctl start docker
    ```

8. **Add Your User to the Docker Group (Optional)**

    ```bash
    sudo usermod -aG docker $USER
    ```

    Log out and back in for the changes to take effect.

9. **Test Docker**
    Run the following command to test Docker:

    ```bash
    sudo docker run hello-world
    ```