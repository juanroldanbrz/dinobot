#!/bin/bash -x

FAKEUSER="${1:-fake-chrome-user}"
CHROMEROOT=$HOME/.chromeroot/

mkdir -p ${CHROMEROOT}

export PROFILE="${CHROMEROOT}/${FAKEUSER}-chromium-profile"
export DISK_CACHEDIR="${CHROMEROOT}/${FAKEUSER}-chromium-profile-cache"
export DISK_CACHESIZE=4096
export MEDIA_CACHESIZE=4096

PARANOID_OPTIONS="\
        --no-displaying-insecure-content \
        --no-referrers \
        --disable-zero-suggest \
        --disable-sync  \
        --cipher-suite-blacklist=0x0004,0x0005,0xc011,0xc007 \
        --enable-sandbox-logging >/dev/null 2>&1
        "


/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
        --ignore-urlfetcher-cert-requests \
        --allow-running-insecure-content \
        --window-position=0,0 \
        --window-size=500,500 \
        --no-pings \
        --user-data-dir=${PROFILE} \
        --disk-cache-dir=${DISK_CACHEDIR} \
        --disk-cache

#GOTO chrome://dino