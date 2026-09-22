#!/bin/bash

# python3 & pip install ruamel.yaml
# istioctl https://gcsweb.istio.io/gcs/istio-build/dev/1.26-alpha.9befed2f1439d883120f8de70fd70d84ca0ebc3d  alpha pre release

GATEWAY_NAMESPACE=llm-d-istio-system

CRD_DIR=deploy/components/crds-istio/
CP_DIR=deploy/components/istio-control-plane/
ISTIO_CP="$(dirname "$0")/istio-cp.yaml"

istioctl manifest generate --dry-run --set values.global.istioNamespace=$GATEWAY_NAMESPACE -f $ISTIO_CP | scripts/istio/manifest-splitter.py -o $CP_DIR
mv $CP_DIR/crds.yaml $CRD_DIR/istio.yaml
