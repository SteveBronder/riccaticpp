#!/usr/bin/env bash
# print_system_info.sh
# Prints a summary of OS, kernel, CPU, cache, and memory specs.

# ---- OS and Kernel ----
echo "=== Operating System ==="
if [ -r /etc/os-release ]; then
  awk -F= '/PRETTY_NAME/ { gsub(/"/, "", $2); print $1 ": " $2 }' /etc/os-release
else
  echo "OS: $(uname -s) (no /etc/os-release)"
fi

echo
echo "=== Kernel ==="
echo "Kernel: $(uname -sr)"

# ---- CPU ----
echo
echo "=== CPU ==="
lscpu | awk '
  /Architecture:/               { print $1, $2 }
  /CPU op-mode/                { print $1 " " $2 ", " $3 }
  /Byte Order:/                { print $1 " " $3 }
  /Model name:/                { $1=""; print "Model: " substr($0,2) }
  /Socket\(s\):/               { print $1, $2 }
  /Core\(s\) per socket:/      { print "Cores/socket:", $4 }
  /Thread\(s\) per core:/      { print "Threads/core:", $4 }
  /CPU MHz:/                   { print "Base MHz:", $3 }
  /CPU max MHz:/               { print "Max MHz:", $4 }
  /CPU min MHz:/               { print "Min MHz:", $4 }
  /Virtualization:/            { print $1 ":", $2 }
  /L1d cache:/                 { print "L1d cache:", $3 }
  /L1i cache:/                 { print "L1i cache:", $3 }
  /L2 cache:/                  { print "L2 cache:", $3 }
  /L3 cache:/                  { print "L3 cache:", $3 }
'

# (Optional) CPU flags — uncomment if you need to audit features
# echo
# echo "=== CPU Flags ==="
# grep -m1 '^flags' /proc/cpuinfo | sed 's/flags\s*: //'

# ---- Memory ----
echo
echo "=== Memory ==="
free -h | awk '
  NR==1 { printf "%-10s %-10s %-10s %-10s\n", $1, $2, $3, $4 }
  NR==2 { printf "%-10s %-10s %-10s %-10s\n", $1, $2, $3, $4 }
'

echo
echo "=== Swap ==="
free -h | awk 'NR==1 || NR==3 { print }'
