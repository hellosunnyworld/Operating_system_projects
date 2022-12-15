#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Page table 32 bit mapping:
// v | dirty | count(18 bits) | vpn (12 bits)
#define COUNTBITS 18
#define VPNBITS   12

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // set invalid(:= MSB is 1 (MSB is the valid bit)) and count = 0b111 (max)
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i; //???
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar * storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

// return valid bit of the pt item
__device__ u32 validBit(u32 ptItem){
  return (ptItem >> 31);
}
// return dirty bit of the pt item
__device__ u32 dirtyBit(u32 ptItem){
  u32 mask = 1 << 30;
  return ((ptItem & mask) >> 30); 
}
// return counting bits of the pt item
__device__ u32 count(u32 ptItem){
  u32 mask = 0x3FFFF000;
  return ((ptItem & mask) >> VPNBITS); 
}
// return vpn of the pt item
__device__ u32 getVpn(u32 ptItem){
  return ((ptItem << (2+COUNTBITS)) >> (2+COUNTBITS));
}
// generate an pt item containing given info
__device__ u32 getPtItem(u32 v, u32 dirty, u32 count, u32 vpn){
  return ((v << 31) + (dirty << 30) + (count << VPNBITS) + vpn);
}

// Transform virtual address to physical address and clear the corresponding count.
// If page fault occurs, increase pagefault_num and load or swap.
__device__ u32 Virtual2Physical(VirtualMemory *vm, u32 virtualAddr){
  u32 vpn = virtualAddr / vm->PAGESIZE;
  u32 ppn = 0;

  // If collision with others
  while ((getVpn(vm->invert_page_table[ppn]) != vpn) && (ppn < vm->PAGE_ENTRIES)) {
    ppn++;
  }

  // page fault: the required page not in pt
  if (ppn >= vm->PAGE_ENTRIES){
     (*(vm->pagefault_num_ptr))++;
     return swap(vm, virtualAddr / vm->PAGESIZE);
  } 
    
  u32 physicalAddr = ppn * vm->PAGESIZE + (virtualAddr % vm->PAGESIZE);
  // Page fault: the page is in pt but invalid
  if (validBit(vm->invert_page_table[ppn]) == 1){
    (*(vm->pagefault_num_ptr))++;
    load(vm, physicalAddr, virtualAddr);
  }

  u32 ptItem;
  u32 c; // current count
  // increase the count of all pt items by 1
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    ptItem = vm->invert_page_table[i];
    c = count(ptItem);
    //printf("origin pt at %x: %x\n", i, ptItem);
    //if (c < ~(1<<COUNTBITS))
    vm->invert_page_table[i] = getPtItem(validBit(ptItem), dirtyBit(ptItem), c + 1, getVpn(ptItem));
    //printf("revised pt at %x: %x\n", i, getPtItem(validBit(ptItem), dirtyBit(ptItem), c + 1, getVpn(ptItem)));
    //printf("valid: %d\n",validBit(ptItem));
    //printf("dirty: %d\n",dirtyBit(ptItem));
    //printf("count: %x\n",c);
    //printf("vpn: %d\n\n\n\n",getVpn(ptItem));
  }
  
  // clear the count of the visited pt item
  ptItem = vm->invert_page_table[ppn];
  vm->invert_page_table[ppn] = getPtItem(validBit(ptItem), dirtyBit(ptItem), 0, getVpn(ptItem));

  // Return physical address
  return physicalAddr;
}

// swap a page into pt and physical memory (if the required page not in pt);
// return physical address
__device__ u32 swap(VirtualMemory *vm, u32 newVpn){
  int leastUsed = 0;
  int maxCount = count(vm->invert_page_table[0]);
  u32 virtualAddr;
  //printf("\n\n\nLRU\n");
  for (int i = 0; i < vm->PAGE_ENTRIES; i++){
    //printf("%d\n", count(vm->invert_page_table[i]));
    if (count(vm->invert_page_table[i]) > maxCount){
      leastUsed = i;
      maxCount = count(vm->invert_page_table[i]);
    }
  }
  // physical address of the page swapped in 
  u32 physicalAddr = leastUsed * vm->PAGESIZE;
  //printf("swap at %d with maxcount %d\n", leastUsed, maxCount);
  // If dirty bit is set
  if (dirtyBit(vm->invert_page_table[leastUsed]) == 1){
    virtualAddr = getVpn(vm->invert_page_table[leastUsed])* vm->PAGESIZE;
    // Write physical memory content to disk (virtual address maps directly to disk address)
    for (int i = 0; i < vm->PAGESIZE; i++)
      vm->storage[virtualAddr + i] = vm->buffer[physicalAddr + i];
  } 

  u32 ptItem;
  u32 c; // current count
  // increase the count of all pt items by 1
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    ptItem = vm->invert_page_table[i];
    c = count(ptItem);
    //printf("origin pt at %x: %x\n", i, ptItem);
    //if (c < ~(1<<COUNTBITS))
    vm->invert_page_table[i] = getPtItem(validBit(ptItem), dirtyBit(ptItem), c + 1, getVpn(ptItem));
    //printf("revised pt at %x: %x\n", i, getPtItem(validBit(ptItem), dirtyBit(ptItem), c + 1, getVpn(ptItem)));
    //printf("valid: %d\n",validBit(ptItem));
    //printf("dirty: %d\n",dirtyBit(ptItem));
    //printf("count: %x\n",c);
    //printf("vpn: %d\n\n\n\n",getVpn(ptItem));
  }
  
  // swap
  virtualAddr = newVpn * vm->PAGESIZE;
  load(vm, physicalAddr, virtualAddr);

  return physicalAddr;
}

// load data from disk to physical memory and update pt
__device__ void load(VirtualMemory *vm, u32 physicalAddr, u32 virtualAddr){
  /* pt update */
  vm->invert_page_table[physicalAddr / vm->PAGESIZE] = getPtItem(0, 0, 0, virtualAddr / vm->PAGESIZE);
  /* physical memory update */
  for (int i = 0; i < vm->PAGESIZE; i++)
    vm->buffer[physicalAddr + i] = vm->storage[virtualAddr + i];
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  return vm->buffer[Virtual2Physical(vm, addr)]; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 physicalAddr = Virtual2Physical(vm, addr);
  u32 ptIndex = physicalAddr / vm->PAGESIZE;

  vm->buffer[physicalAddr] = value;
  // set the dirty bit in pt
  vm->invert_page_table[ptIndex] = getPtItem(0, 1, 0, getVpn(vm->invert_page_table[ptIndex]));
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = offset; i < offset + input_size; i++){
    results[i] = vm_read(vm, i);
  }
}

