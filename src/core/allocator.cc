#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
       for (auto it = freeByAddr.begin(); it != freeByAddr.end(); ++it)
        {
            if (it->second >= size)
            {
                size_t addr = it->first;
                size_t blockSize = it->second;
                size_t blockEnd = addr + blockSize;
                freeByAddr.erase(it);
                freeByEnd.erase(blockEnd);
                if (blockSize > size)
                {
                    size_t newAddr = addr + size;
                    size_t newSize = blockSize - size;
                    freeByAddr[newAddr] = newSize;
                    freeByEnd[newAddr + newSize] = newAddr;
                }
                return addr;
            }
        }

        size_t addr = used;
        used += size;
        peak = std::max(peak, used);
        return addr;
        // return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t start = addr;
        size_t end = addr + size;

        auto leftIt = freeByEnd.find(start);
        if (leftIt != freeByEnd.end())
        {
            size_t leftStart = leftIt->second;
            size_t leftSize = start - leftStart;
            freeByEnd.erase(leftIt);
            freeByAddr.erase(leftStart);
            start = leftStart;
            size += leftSize;
            end = start + size;
        }

        auto rightIt = freeByAddr.find(end);
        if (rightIt != freeByAddr.end())
        {
            size_t rightStart = rightIt->first;
            size_t rightSize = rightIt->second;
            size_t rightEnd = rightStart + rightSize;
            freeByAddr.erase(rightIt);
            freeByEnd.erase(rightEnd);
            size += rightSize;
            end = start + size;
        }

        if (end == used)
        {
            used = start;
            while (true)
            {
                auto it = freeByEnd.find(used);
                if (it == freeByEnd.end())
                {
                    break;
                }
                size_t prevStart = it->second;
                freeByEnd.erase(it);
                freeByAddr.erase(prevStart);
                used = prevStart;
            }
            return;
        }

        freeByAddr[start] = size;
        freeByEnd[start + size] = start;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
