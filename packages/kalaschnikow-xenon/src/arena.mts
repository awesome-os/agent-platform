/**
 * @module
 * B8G Zero-Copy Memory Arena (`B8G_ARENA_V1`) for Model Kalaschnikow.
 *
 * Implements a flat, contiguous binary memory arena backed by `SharedArrayBuffer` / `Uint8Array`.
 * Supports zero-allocation sub-view indexing, parameter registration, cache-line alignment (64 bytes),
 * and binary serialization compatible with zero-overhead mmap and WASM/V8 runtime ingestion.
 */

import * as fs from 'node:fs'
import type { ArenaTensorDescriptor, ArenaHeader } from './types.mts'

export const ARENA_MAGIC = 'B8G_ARENA_V1'
export const ARENA_ALIGNMENT = 64 // 64-byte CPU cache line alignment

export class B8GModelArena {
  private readonly buffer: SharedArrayBuffer
  private readonly byteView: Uint8Array
  private readonly tensorMap = new Map<string, { descriptor: ArenaTensorDescriptor; view: Float32Array }>()
  private writeOffset: number = 0

  constructor(totalByteCapacity: number = 512 * 1024 * 1024) {
    this.buffer = new SharedArrayBuffer(totalByteCapacity)
    this.byteView = new Uint8Array(this.buffer)
  }

  /**
   * Returns the underlying SharedArrayBuffer.
   */
  public getSharedArrayBuffer(): SharedArrayBuffer {
    return this.buffer
  }

  /**
   * Returns current allocated byte size in arena.
   */
  public getUsedBytes(): number {
    return this.writeOffset
  }

  /**
   * Returns list of registered tensor descriptors.
   */
  public getTensorDescriptors(): ArenaTensorDescriptor[] {
    return Array.from(this.tensorMap.values()).map(v => v.descriptor)
  }

  /**
   * Allocates or registers a tensor in contiguous linear memory with 64-byte alignment.
   */
  public allocateTensor(name: string, shape: readonly number[], initialData?: Float32Array): Float32Array {
    if (this.tensorMap.has(name)) {
      return this.tensorMap.get(name)!.view
    }

    const elementCount = shape.reduce((a, b) => a * b, 1)
    const byteLength = elementCount * 4 // float32

    // Align offset to 64 bytes
    const alignedOffset = (this.writeOffset + (ARENA_ALIGNMENT - 1)) & ~(ARENA_ALIGNMENT - 1)
    if (alignedOffset + byteLength > this.buffer.byteLength) {
      throw new Error(`B8GModelArena: Out of memory in arena. Capacity: ${this.buffer.byteLength}, required: ${alignedOffset + byteLength}`)
    }

    const view = new Float32Array(this.buffer, alignedOffset, elementCount)
    if (initialData) {
      view.set(initialData)
    }

    const descriptor: ArenaTensorDescriptor = {
      name,
      shape,
      dtype: 'float32',
      offsetBytes: alignedOffset,
      byteLength,
      elementCount
    }

    this.tensorMap.set(name, { descriptor, view })
    this.writeOffset = alignedOffset + byteLength
    return view
  }

  /**
   * Retrieves zero-copy tensor view.
   */
  public getTensor(name: string): Float32Array | undefined {
    return this.tensorMap.get(name)?.view
  }

  /**
   * Checks if tensor exists in arena.
   */
  public hasTensor(name: string): boolean {
    return this.tensorMap.has(name)
  }

  /**
   * Serializes all arena tensors into the standard B8G_ARENA_V1 binary layout:
   * [12 bytes] Magic: "B8G_ARENA_V1"
   * [4 bytes]  Tensor count (UInt32 LE)
   * For each tensor:
   *   [2 bytes] Name length N (UInt16 LE)
   *   [N bytes] Name UTF-8 bytes
   *   [1 byte]  ndim (UInt8)
   *   [8 * ndim bytes] shape dimensions (UInt64 LE)
   *   [8 bytes] byte length of tensor data (UInt64 LE)
   *   [data bytes] raw Float32 LE buffer
   */
  public serializeToBinary(): Buffer {
    const magicBuf = Buffer.from(ARENA_MAGIC, 'ascii')
    const countBuf = Buffer.alloc(4)
    countBuf.writeUInt32LE(this.tensorMap.size, 0)

    const chunks: Buffer[] = [magicBuf, countBuf]

    for (const [name, { descriptor, view }] of this.tensorMap.entries()) {
      const nameBuf = Buffer.from(name, 'utf-8')
      const headerBuf = Buffer.alloc(2 + nameBuf.length + 1 + descriptor.shape.length * 8 + 8)
      let offset = 0

      // Name length
      headerBuf.writeUInt16LE(nameBuf.length, offset)
      offset += 2

      // Name string
      nameBuf.copy(headerBuf, offset)
      offset += nameBuf.length

      // ndim
      headerBuf.writeUInt8(descriptor.shape.length, offset)
      offset += 1

      // Dimensions (BigUInt64)
      for (const dim of descriptor.shape) {
        headerBuf.writeBigUInt64LE(BigInt(dim), offset)
        offset += 8
      }

      // Tensor byte length
      headerBuf.writeBigUInt64LE(BigInt(descriptor.byteLength), offset)
      offset += 8

      chunks.push(headerBuf)

      // Tensor data buffer (Float32Array bytes)
      const dataBuf = Buffer.from(view.buffer, view.byteOffset, view.byteLength)
      chunks.push(dataBuf)
    }

    return Buffer.concat(chunks)
  }

  /**
   * Saves serialized arena to disk in a single write operation.
   */
  public saveToFile(filePath: string): void {
    const binary = this.serializeToBinary()
    fs.writeFileSync(filePath, binary)
  }

  /**
   * Loads arena from binary B8G_ARENA_V1 file into this zero-copy arena.
   */
  public static loadFromFile(filePath: string, bufferCapacity?: number): B8GModelArena {
    const fileBuf = fs.readFileSync(filePath)
    return B8GModelArena.deserialize(fileBuf, bufferCapacity)
  }

  /**
   * Deserializes B8G_ARENA_V1 binary into a fresh B8GModelArena.
   */
  public static deserialize(buffer: Buffer, bufferCapacity?: number): B8GModelArena {
    const magic = buffer.subarray(0, 12).toString('ascii')
    if (magic !== ARENA_MAGIC) {
      throw new Error(`Invalid arena magic: expected '${ARENA_MAGIC}', got '${magic}'`)
    }

    let offset = 12
    const tensorCount = buffer.readUInt32LE(offset)
    offset += 4

    // Determine required capacity
    const estimatedNeeded = buffer.length * 2
    const capacity = bufferCapacity ?? Math.max(estimatedNeeded, 128 * 1024 * 1024)
    const arena = new B8GModelArena(capacity)

    for (let i = 0; i < tensorCount; i++) {
      const nameLen = buffer.readUInt16LE(offset)
      offset += 2

      const name = buffer.subarray(offset, offset + nameLen).toString('utf-8')
      offset += nameLen

      const ndim = buffer.readUInt8(offset)
      offset += 1

      const shape: number[] = []
      for (let d = 0; d < ndim; d++) {
        shape.push(Number(buffer.readBigUInt64LE(offset)))
        offset += 8
      }

      const byteLen = Number(buffer.readBigUInt64LE(offset))
      offset += 8

      // Ensure 4-byte / 64-byte alignment when extracting Float32Array from binary slice
      const floatCount = byteLen / 4
      const alignedBuffer = new ArrayBuffer(byteLen)
      new Uint8Array(alignedBuffer).set(buffer.subarray(offset, offset + byteLen))
      const floatArray = new Float32Array(alignedBuffer, 0, floatCount)
      offset += byteLen

      arena.allocateTensor(name, shape, floatArray)
    }

    return arena
  }
}
