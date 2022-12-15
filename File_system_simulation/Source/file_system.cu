#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

__device__ __managed__ uint64_t gtime = 0; // global initial time
__device__ u32 addTimes;

  
__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  fileNum = 0;
  addTimes = 0;
  // init volume control block and FCB
  for (int i = 0; i < 36*1024; i++)
    fs->volume[i] = 0x00; // means NULL
  gtime = 0;
}

// Write 1s to volume control block when allocating blocks, whose number is num, starting at startIndex 
__device__ void RecordOccupied(FileSystem *fs, int startIndex, int num){
  u32 start = startIndex % 8;
  int entryNum = (start + num) / 8 + ((start + num) % 8 != 0);
  uint64_t writeData;
  if (num < 32)
    writeData = ((1 << num) - 1) << (8*entryNum - start - num);
  else
    writeData = 0xFFFFFFFF << (8*entryNum - start - num);    
  uint8_t mask;
  for (int i = 0; i < entryNum; i++){
    mask = (writeData << (8*i)) >> (8*(entryNum - 1));
    //printf("modify %d th bit map %x with mask %x...\n", startIndex/8 + i, fs->volume[startIndex/8 + i], mask);
    fs->volume[startIndex/8 + i] |= mask;
    //printf("have modified %d th bit map %x\n\n", startIndex/8 + i, fs->volume[startIndex/8 + i]);
  }
}
__device__ void ClearOccupied(FileSystem *fs, int startIndex, int num){
  u32 start = startIndex % 8;
  uint64_t writeData;
  int entryNum = (start + num) / 8 + ((start + num) % 8 != 0);
  if (num < 32)
    writeData = ((1 << num) - 1) << (8*entryNum - start - num);
  else
    writeData = 0xFFFFFFFF << (8*entryNum - start - num);  
  uint8_t mask;
  for (int i = 0; i < entryNum; i++){
    mask = (writeData << (8*i)) >> (8*(entryNum - 1));
    //printf("clear %d th bit map %x with mask %x...\n", startIndex/8 + i, fs->volume[startIndex/8 + i], mask);
    fs->volume[startIndex/8 + i] &= ~mask;
    //printf("have cleared %d th bit map %x\n\n", startIndex/8 + i, fs->volume[startIndex/8 + i]);
  }
}
__device__ u32 LocateSpace(FileSystem *fs, int size){
  // size is the number of required blocks
  // return the index of the empty block
  uint64_t mask;
  //printf("locate space size = %d\n", size);
  if (size < 32)
      mask = (1 << size) - 1;
  else
      mask = 0xFFFFFFFF;
  //printf("mask: %x\n", mask);
  uint64_t map;
  for (u32 i = 0; i < (fs->SUPERBLOCK_SIZE - 4); i++){
    map = ((uint64_t)fs->volume[i] << 32) + ((uint64_t)fs->volume[i+1] << 24) + ((uint64_t)fs->volume[i+2] << 16) + ((uint64_t)fs->volume[i+3] << 8) + (uint64_t)fs->volume[i+4];
    //printf("map %d: %llx\n", i, map);
    if ((uint64_t)(0xFFFFFFFFFF - map) < mask){
           //printf("continue\n");
      continue;
    }
    else{
      for (int j = 0; j < (40-size+1); j++){
        //printf("locate space size = %d\n", size);
        //printf("40-size-j = %d\n",40-size-j);
        uint64_t mask_t = mask << (40-size-j);
        uint64_t checkResult = ((map | (~mask_t)) << (24+j)) >> (64-size);
        //printf("mask' %llx\n", mask_t);
        //printf("check result = %llx\n", checkResult);
        //printf("------------\n");
        if (checkResult == 0){
        //if (map & (mask << (32-size-j)) == 0){
          //printf("i = %d\n", i);
          //printf("j = %d\n", j);
          //printf("locate at %dth block\n", 8*i+j);
          return (8*i+j);
      }
    }      
  }
 }

  /*else{
    int x,y;
    mask = 0xFFFFFFFF;
    for (u32 i = 0; i < (fs->SUPERBLOCK_SIZE - 1); i++){
      if ((fs->volume[i+1] != 0x00) || (fs->volume[i+2] != 0x00)|| (fs->volume[i+3] != 0x00))
          continue;
          
      x = 1;          
      while (fs->volume[i] % (1<<x) == 0)
        x++;
      x--;
      
      y = 1;
      while (fs->volume[i+4] >> (8-y) == 0)
        y++;
      y--;  
      printf("x,y = %d %d\n", x, y);
      if (x+y >= 8)
        return (8*i+8-x);   
    }
  }  */
}

// Search the given file in the FCB. If it is not in FCB, add its name into FCB.
// Return the base address of the corresponding FCB entry in volume
__device__ u32 search_FCB_entry(FileSystem *fs, char* s, int op){
  //printf("%s",s);
  u32 baseAddr;
  int nullNum = 0;
  int delNum = 0; // the number of "DELETED" tags
  u32 delAddr = 0xFFFFFFFF;
  uchar name[1];
  bool flag = 0;
  int t = 0;
      /*while (s[t] != '\0'){
        name[0] = s[t];
        printf("%s",name); 
        t++;      
      }*/
  t = 0;
  for (int i = 0; i < fs->FCB_ENTRIES; i++){
    t = 0;
    baseAddr = 1024*4 + 32 * i;
    while (s[t] != '\0'){
      // the file is not in this entry 
      if (fs->volume[baseAddr + t] != s[t]){
        //printf("%d %d: the file %s is not in this entry\n",i, t, s);
        break;
      }
      t++;
    }
    // this entry records the given file
    if ((s[t] == '\0') && (fs->volume[baseAddr + t] == '\0')){
        //printf("this entry records the given file\n");
        return baseAddr;
    }  

    for (int j = 0; j < 20; j++){
      if (fs->volume[baseAddr + j] == 0xFF)
        delNum++;
      else if (fs->volume[baseAddr + j] != 0x00){
        flag = 1;
        break;
      }
      else
        nullNum++;
      //printf("delNum:%d\n",delNum);
    }
    if (flag){
      flag = 0;
      continue;
    }
    if (delNum == 20){
      delNum = 0;
      if (delAddr == 0xFFFFFFFF)
        delAddr = baseAddr;
      continue;
    }
    // the file is not in FCB
    // If to read the file, report an error
    if (op == G_READ){
      //printf("search FCB %d\n", i);
      printf("Error: Cannot find the file\n");
      for (int i = 4*1024; i < 36*1024; i+=32){
        //printf("%s\n",checkName(fs,i));
      }
      //printf("search for %s\n",s);
      return 0xFFFFFFFF;
    }
    // If to write the file, create a new file and add it into FCB
    t = 0;
    //printf("%d\n",i == (fs->FCB_ENTRIES - 1));
    // this entry is empty
    if ((nullNum == 20) || (i == (fs->FCB_ENTRIES - 1))){ 
    //if ((nullNum == 20)){ 
      //printf("add!!!\n");
      if ((i == (fs->FCB_ENTRIES - 1)) && (delAddr != 0xFFFFFFFF))
        baseAddr = delAddr;
      fileNum++;
      addTimes++;
      // have achieved the empty part of FCB: the given file is new
      // add the file name into FCB
      while (s[t] != '\0'){
        fs->volume[baseAddr + t] = s[t];
        name[0] = fs->volume[baseAddr + t];
        //printf("%s",name);  
        //name[0] = s[t];
        //printf("%s",name);
        t++;       
      }
      if (delAddr != 0xFFFFFFFF){
        for (int k = t; k < 20; k++){
          fs->volume[baseAddr + k] = 0x00;
        }
      }
      //printf("\n");
      modifyStartingAddr(fs, baseAddr, 0xFFFF); //0xFFFF means no block allocated to it
      modify_ModifyT(fs, baseAddr, gtime);
      modifySize(fs, baseAddr, 0);
      //printf("after add\n");
      return baseAddr;
    }
    nullNum = 0;
    delNum = 0;
  }
}
__device__ u32 checkSize(FileSystem *fs, u32 fp){
  if (fs->volume[fp + 31] == 0x01)
    return 1024;
  else{
    uint8_t high = fs->volume[fp + 20];
    uint8_t low = fs->volume[fp + 21] >> 6;
    return (high << 2) + low;
  }
}
__device__ void modifySize(FileSystem *fs, u32 fp, u32 newSize){
  if (newSize == 1024)
    fs->volume[fp + 31] = 0x01;
  else{
    uint8_t high = newSize >> 2;
    fs->volume[fp + 20] = high;
    uint8_t low = (newSize << 30) >> 24; // low000000
    fs->volume[fp + 21] &= 0x3F; // clear upper 2 bits
    fs->volume[fp + 21] |= low;
  }
}
__device__ u32 checkStartingAddr(FileSystem *fs, u32 fp){
  uint8_t high = (uint8_t)(fs->volume[fp + 21] << 2) >> 2;
  //printf("fs->volume[fp + 21] = %x\n",fs->volume[fp + 21]);
  //printf("fs->volume[fp + 21] << 2 = %x\n",(uint8_t)(fs->volume[fp + 21] << 2));
  //printf("read high = %x\n", high);
  uint8_t mid = fs->volume[fp + 22];
  //printf("read mid = %x\n", mid);
  uint8_t low = fs->volume[fp + 23] >> 7;
  //printf("read low = %x\n", low);
  return 36*1024+(((high << 9) + (mid << 1) + low) << 5);  
}
__device__ void modifyStartingAddr(FileSystem *fs, u32 fp, u32 newAddr){
  newAddr = ((newAddr-36*1024) << 12) >> 17;
  uint8_t high = newAddr >> 9;
  //printf("high: %x\n", high);
  fs->volume[fp + 21] &= 0xC0; // clear the lower 6 bits
  fs->volume[fp + 21] |= high;
  uint8_t mid = (newAddr << 23) >> 24; // mid
  fs->volume[fp + 22] = mid;
  //printf("mid: %x\n", mid);
  uint8_t low = (newAddr << 31) >> 24; // low0000000
  //printf("before clear MSB:%x\n", fs->volume[fp + 23]); 
  fs->volume[fp + 23] &= 0x7F; // clear MSB
  //printf("cleared MSB:%x\n", fs->volume[fp + 23]); 
  fs->volume[fp + 23] |= low; 
  //printf("added low:%x\n", fs->volume[fp + 23]); 
  //printf("low: %x\n", low);
}
__device__ uint64_t checkModifyT(FileSystem *fs, u32 fp){
  //printf("raw high time: %x\n", fs->volume[fp + 23]);
  //uint8_t high0 = (fs->volume[fp + 23] << 1);
  //uint8_t high = high0 >> 1;
  //printf("high time: %x %x\n", fs->volume[fp + 23] << 1, high);
  uint64_t top = fs->volume[fp + 24];
  uint64_t high = fs->volume[fp + 25];
  uint64_t mid = fs->volume[fp + 26];
  uint64_t low = fs->volume[fp + 27];
  uint64_t bottom = fs->volume[fp + 28];
  return (uint64_t)(top << 32) + (uint64_t)(high << 24) + (uint64_t)(mid << 16) + (uint64_t)(low << 8) + (uint64_t)bottom; 
}
__device__ void modify_ModifyT(FileSystem *fs, u32 fp, uint64_t newT){
  uint8_t top = (newT << 24) >> 56;
  //fs->volume[fp + 23] &= 0x80; // clear the lower 7 bits
  //fs->volume[fp + 23] |= high;
  fs->volume[fp + 24] = top;
  uint8_t high = (newT << 32) >> 56;
  fs->volume[fp + 25] = high;
  uint8_t mid = (newT << 40) >> 56;
  fs->volume[fp + 26] = mid;
  uint8_t low = (newT << 48) >> 56; // low
  fs->volume[fp + 27] = low;
  uint8_t bottom = (newT << 56) >> 56;
  fs->volume[fp + 28] = bottom;
  //printf("modify time to %llx\n",newT);
}
__device__ uchar* checkName(FileSystem *fs, u32 fp){
  uchar name[20];
  int i = 0;
  while (fs->volume[fp + i] != '\0'){
    name[i] = fs->volume[fp + i];
    i++;
  }
  name[i] = '\0';
  return name;
}
__device__ u32 checkCreateT(FileSystem *fs, u32 fp){
  //return (fs->volume[fp + 27] << 8 + fs->volume[fp + 28]);
}
__device__ void modifyCreateT(FileSystem *fs, u32 fp){
  //fs->volume[fp + 27] = (addTimes << 16) >> 24;
  //fs->volume[fp + 28] = (addTimes << 24) >> 24;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  //printf("open %s\n", s);
  gtime++;
	u32 fp =  search_FCB_entry(fs, s, op);
  //printf("get a fp:%d\n", fp);
  if (op == G_WRITE){
    //printf("check size...\n");
    int size = checkSize(fs, fp);
    //printf("size=%d\n", size);
    if (size > 0){
      //printf("clear for write\n");
      u32 addr = checkStartingAddr(fs, fp);
      //printf("should open at addr %x\n", addr);
      for (int i = 0; i < size; i++){
        fs->volume[addr + i] = 0x00;
      }
    }
    uint64_t time = gtime;
    //modify_ModifyT(fs, fp, gtime);
    modify_ModifyT(fs, fp, time);
    //printf("modify time to %x...\n",time);
    //printf("modified time=%x\n", checkModifyT(fs, fp));
  }
  //printf("addTimes=%d %d\n",addTimes, checkCreateT(fs,fp));
  //printf("open return\n");
  return fp;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  gtime++;
  //printf("read\n");
	int fsize = checkSize(fs, fp);
  if (fsize != 0){ // there is content in this file
    u32 addr = checkStartingAddr(fs, fp);
    // if the required size is larger than the file size, read out the whole file 
    if (size > fsize) 
      size = fsize;
    for (int i = 0; i < size; i++){
      output[i] = fs->volume[addr + i];
    }
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
  //printf("write\n");
  gtime++;
	int fsize = checkSize(fs, fp);
  int newBlockSize = size / 32 + (size % 32 != 0);
  //printf("newBlockSize = %d\n", newBlockSize);
  int oldBlockSize = fsize / 32 + (fsize % 32 != 0); // the file size must be integer number of blocks
  int oldStartBlock = (checkStartingAddr(fs, fp) - fs->FILE_BASE_ADDRESS) / 32;
  //printf("new time %x of fp %d\n", checkModifyT(fs, fp), fp);
  /* Switch the file size */
  // Update file size in FCB
  //printf("Update file size in FCB\n");
  modifySize(fs, fp, size);
  if ((fsize > 0) && (newBlockSize != oldBlockSize))
    ClearOccupied(fs, oldStartBlock, oldBlockSize);
  if (newBlockSize > oldBlockSize){
    compact(fs, fp);
    /* Enlarge the file : Move to a larger space */
    u32 newStartBlock = LocateSpace(fs, newBlockSize);
    //printf("Locate at %u th block with length %d\n", newStartBlock, newBlockSize);
    // Update bit map
    RecordOccupied(fs, newStartBlock, newBlockSize);
    //printf("modify addr to %x ...\n", fs->FILE_BASE_ADDRESS + 32*newStartBlock);
    //printf("time: %x\n", checkModifyT(fs,fp));
    // Update starting addr in FCB
    modifyStartingAddr(fs, fp, fs->FILE_BASE_ADDRESS + 32*newStartBlock);
    //printf("have modified addr to %x\n", checkStartingAddr(fs, fp));
    //printf("time: %x\n", checkModifyT(fs,fp));
  }
  else if (newBlockSize < oldBlockSize){
    /* Shrink the file */
    // Update bit map
    RecordOccupied(fs, oldStartBlock, newBlockSize);
  }

  // Write to the file
  u32 addr = checkStartingAddr(fs, fp);
  //printf("Write to the file at %x\n", addr);
  for (u32 i = 0; i < size; i++){
    //printf("write %dth byte to %x...\n", i, addr + i);
    fs->volume[addr + i] = input[i];
  }
  //printf("finish\n");
  
  // Update modification time
  //printf("Update modification time\n");
  uint64_t time = gtime;
  //printf("current time: %llx\n", time);
  modify_ModifyT(fs, fp, time);
  //printf("new time %u of fp %d\n", checkModifyT(fs, fp), fp);
  //printf("write return\n");
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  u32 records[1024];
  bool swap;
  u32 item;
  int j;
  int delPos = 1; // the index of the currently visited DELETED entry which contains file
  u32 fp = 1024*4 - 32;
  // Record the files into an array
  for (u32 i = 0; i < fileNum; i++){
    fp += 32;
    while (fs->volume[fp] == 0xFF){
      fp += 32;
    }
    records[i] = fp;
    //printf("item %d: %s\n", 0, records[0]->name);
    //printf("exchange with %d: %s\n", 1, records[1]->name);
  }
  // Sort the array by insertion sort
  for (int i = 1; i < fileNum; i++){
    if (op == LS_D)
      swap = (checkModifyT(fs, records[i]) > checkModifyT(fs, records[i-1]));
    else if (op == LS_S){
      swap = (checkSize(fs, records[i]) > checkSize(fs, records[i-1]));
      if (checkSize(fs, records[i]) == checkSize(fs, records[i-1]))
        swap = (checkModifyT(fs, records[i]) > checkModifyT(fs, records[i-1]));
        //swap = (checkCreateT(fs, records[i]) < checkCreateT(fs, records[i-1]));
    }
    j = i;
    while ((swap) && (j > 0)){
      item = records[j-1];          
      records[j-1] = records[j];
      records[j] = item;
      j--;
      if (j > 0){
        if (op == LS_D)
          swap = (checkModifyT(fs, records[j]) > checkModifyT(fs, records[j-1]));
        else if (op == LS_S){
          swap = (checkSize(fs, records[j]) > checkSize(fs, records[j-1]));
          if (checkSize(fs, records[j]) == checkSize(fs, records[j-1]))
            swap = (checkModifyT(fs, records[j]) > checkModifyT(fs, records[j-1]));
            //swap = (checkCreateT(fs, records[j]) < checkCreateT(fs, records[j-1]));
        }
      }
      else
        break;
    }
  }
  
  if (op == LS_D){
    printf("===sort by modified time===\n");
    for (int i = 0; i < fileNum; i++){
      //printf("%s %u\n", checkName(fs, records[i]), checkModifyT(fs, records[i]));
      printf("%s\n", checkName(fs, records[i]));
    }
  }
  else if (op == LS_S){
    printf("===sort by file size===\n");
    for (int i = 0; i < fileNum; i++){
      printf("%s %u\n", checkName(fs, records[i]), checkSize(fs, records[i]));
      //printf("%s %u %llx\n", checkName(fs, records[i]), checkSize(fs, records[i]), checkModifyT(fs, records[i]));
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	u32 fp =  search_FCB_entry(fs, s, op);
  u32 addr = checkStartingAddr(fs, fp);
  int blockIndex = (addr - fs->FILE_BASE_ADDRESS) / 32;
  int size = checkSize(fs, fp);
  // clear bit map
  ClearOccupied(fs, blockIndex, size/32 + (size%32!=0));
  // clear FCB
  for (int i = 0; i < 32; i++){
    fs->volume[fp + i] = 0xFF; // means "DELETED"
  }
  // clear storage
  for (int i = 0; i < 32*size; i++){
    fs->volume[addr + i] = 0x00;
  }  
  fileNum--;
  compact(fs, fp);
}

// compact all files starting from the fp of the new space in storage
__device__ void compact(FileSystem *fs, u32 fp){
  /*for (int i = fp; i < (4*1024+992); i++){
      fs->volume[i] = fs->volume[i + 32] ;
  }*/
  //printf("\ncompact\n-------------------------------\n");
 	int fsize;
  int oldBlockSize; // the file size must be integer number of blocks
  int oldStartBlock;
  u32 newStartBlock;
  u32 oldAddr;
  u32 addr;
  for (int i = fp+32; i < 36*1024; i+=32){
    while (fs->volume[i] == 0xFF){
      i += 32;
      if (i >= 36*1024)
        break;
    } 
    if (i >= 36*1024)
      break;
      
   	fsize = checkSize(fs, i);
    if (fsize > 0){
      oldBlockSize = fsize / 32 + (fsize % 32 != 0); // the file size must be integer number of blocks
      oldAddr = checkStartingAddr(fs, i);
      oldStartBlock = (oldAddr - fs->FILE_BASE_ADDRESS) / 32;
      
      ClearOccupied(fs, oldStartBlock, oldBlockSize);
      /* Enlarge the file : Move to a larger space */
      newStartBlock = LocateSpace(fs, oldBlockSize);
      //printf("Locate at %u th block with length %d\n", newStartBlock, oldBlockSize);
      // Update bit map
      RecordOccupied(fs, newStartBlock, oldBlockSize);
      //printf("modify addr to %x ...\n", fs->FILE_BASE_ADDRESS + 32*newStartBlock);
      //printf("time: %x\n", checkModifyT(fs,fp));
      // Update starting addr in FCB
      modifyStartingAddr(fs, i, fs->FILE_BASE_ADDRESS + 32*newStartBlock); 
      
      // Write to the storage
      addr = checkStartingAddr(fs, i);
      //printf("Write to the file at %x\n", addr);
      for (u32 j = 0; j < fsize; j++){
        //printf("write %dth byte to %x...\n", i, addr + i);
        fs->volume[addr + j] = fs->volume[oldAddr+j];
      }
      //show(fs,i);
    }
  }
  //printf("-------------------------------\ncompact return\n\n");
}

__device__ void show(FileSystem *fs, u32 fp){
      //printf("show!\n");
      //printf("fp %d\n", fp); 
      printf("name: %s\n", checkName(fs, fp));
      printf("size: %d\n", checkSize(fs, fp));
      printf("start address: %x\n", checkStartingAddr(fs, fp)); //0xFFFF means no block allocated to it
      printf("time: %llx\n", checkModifyT(fs, fp)); 
      printf("\n");
}