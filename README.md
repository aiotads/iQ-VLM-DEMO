# iQ-VLM-DEMO

<img src="docs/vlm.gif" width="500"/>

## Prerequisites

* A running **OGenie server**
* A **webcam** connected to the **Qualcomm platform**

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
