set -e

DROIDCAM_IP=$(python3 /home/skippy/scripts/get_android_ip.py)
/home/skippy/droidcam/droidcam-cli $DROIDCAM_IP 4747