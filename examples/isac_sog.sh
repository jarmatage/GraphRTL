#!/bin/bash
for prefix in $(ls ${RTL_DIR}/*.v | xargs -n 1 -I {} basename {} .v); do
    echo "Running yosys on $prefix"
    graphrtl yosys -o . ${prefix}_bench ${RTL_DIR}/${prefix}.v
done
