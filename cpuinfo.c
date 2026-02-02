#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  // CPU name
  system("cat /proc/cpuinfo | grep name -m1");

  // CPU count
  system("cat /proc/cpuinfo | grep processor | tail -n1");

  // CPU freq
  system("cat /proc/cpuinfo | grep \"cpu MHz\"");

  // RAM
  system("free -h | grep Mem");

  return 0;
}
