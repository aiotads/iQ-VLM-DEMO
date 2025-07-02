# iQ-VLM-DEMO

<img src="docs/vlm.gif" width="500"/>

## Prerequisites

* A running **OGenie server**.
  > not publicly available, please contact us for access
* A **webcam** connected to the **Qualcomm platform**. (1920x1080 MJPG 30 FPS)

## Steps on the Qualcomm Platform

1. **Clone the repository**

   ```bash
   git clone https://github.com/aiotads/iQ-VLM-DEMO.git
   cd iQ-VLM-DEMO
   ```

2. **Build the Docker image**

   ```bash
   docker build -t iq-vlm-demo -f docker/Dockerfile ./
   ```

3. **Run the demo**

   ```bash
   ./docker/run.sh
   ```

   <img src="docs/console.png" width="600"/>

4. **Open `192.168.3.206:4000` using your web browser**
