#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

extern __device__ int fileNum; // Record the number of files in FCB
extern __device__ u32 addTimes;
//extern __device__ int delPosNum;

struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

// volume control block functions
__device__ void RecordOccupied(FileSystem *fs, int startIndex, int num);
__device__ void ClearOccupied(FileSystem *fs, int startIndex, int num);
__device__ u32 LocateSpace(FileSystem *fs, int size);

// FCB functions
__device__ u32 search_FCB_entry(FileSystem *fs, char* s, int op);
__device__ u32 checkSize(FileSystem *fs, u32 fp);
__device__ void modifySize(FileSystem *fs, u32 fp, u32 newSize);
__device__ u32 checkStartingAddr(FileSystem *fs, u32 fp);
__device__ void modifyStartingAddr(FileSystem *fs, u32 fp, u32 newAddr);
__device__ uint64_t checkModifyT(FileSystem *fs, u32 fp);
__device__ void modify_ModifyT(FileSystem *fs, u32 fp, uint64_t newT);
__device__ uchar* checkName(FileSystem *fs, u32 fp);
__device__ void compact(FileSystem *fs, u32 fp);
__device__ int checkEmpty(FileSystem *fs, u32 fp);

__device__ void show(FileSystem *fs, u32 fp);
#endif