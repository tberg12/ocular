package edu.berkeley.cs.nlp.ocular.lm;

import java.util.Arrays;

public class CountDbSimple implements CountDb {
  public long[] keys;
  public int[][] countVals;
  public final int numCountTypes;
  private long trainNumTokens;
  private int trainNumBigramTypes;
  private int numEntries;
  private int numProbes;
  private int numQueries;
  
  public CountDbSimple(int numKeys, int numCountTypes) {
    this.keys = new long[numKeys];
    Arrays.fill(keys, 0);
    int totalNumCountTypes = numCountTypes;
    this.countVals = new int[totalNumCountTypes][numKeys];
    for (int i = 0; i < totalNumCountTypes; i++) {
      for (int j = 0; j < numKeys; j++) {
        countVals[i][j] = 0;
      }
    }
    this.numCountTypes = numCountTypes;
    this.trainNumBigramTypes = 0;
    this.numEntries = 0;
    this.numProbes = 0;
    this.numQueries = 0;
  }
  
  public long getNumTokens() {
    return trainNumTokens;
  }
  
  public int getNumBigramTypes() {
    return trainNumBigramTypes;
  }
  
  public int currSize() {
    return numEntries;
  }
  
  public int totalSize() {
    return countVals[0].length;
  }

  @Override
  public long[] getKeys() {
    return keys;
  }
  
  public int getCount(NgramWrapper ngram, CountType countType) {
    return this.getCount(ngram.getLongRep(), countType);
  }
  
  public int getCount(long ngram, CountType countType) {
    int countTypeIndex = countType.getIndex();
    return countVals[countTypeIndex][find(ngram)];
  }
  
  public void incrementBigramTypes() {
    trainNumBigramTypes++;
  }

  public int incrementCount(NgramWrapper ngram, CountType countType) {
    return this.incrementCount(ngram.getLongRep(), countType);
  }
  
  /**
   * @param ngram
   * @param countType
   * @param trainTestSelector
   * @return The old count of the ngram (pre-update), but only if we do token counts
   */
  public int incrementCount(long ngram, CountType countType) {
    int countTypeIndex = countType.getIndex();
    int index = find(ngram);
    int oldCount = countVals[countTypeIndex][index];
    if (keys[index] != ngram)
      numEntries++;
    if (countType == CountType.TOKEN_INDEX) {
      trainNumTokens++;
    }
    keys[index] = ngram;
    countVals[countTypeIndex][index]++;
    return oldCount;
  }
  
  private int find(long key) {
    int hashToArray = ((int)(key^(key>>>32)) * 3875239) % totalSize();
    if (hashToArray < 0)
      hashToArray += totalSize();
    numQueries++;
    numProbes++;
    // Until we find the key or a blank space to put it in
    while (keys[hashToArray] != key && keys[hashToArray] != 0)
    {
      numProbes++;
      hashToArray = (hashToArray + 1) % totalSize();
    }
    return hashToArray;
  }
  
  public void maybeResize() {
    if (numEntries * 1.08 > countVals[0].length)
    {
      if (Runtime.getRuntime().freeMemory() < countVals[0].length * (2 + countVals.length) * 4 * 0.6)
      {
        System.out.println("WARNING: need more than " + Runtime.getRuntime().freeMemory()/(1024*1024)
                     + " MB in order to expand");
      }
      // Resize additively because because at this size, it will get too big otherwise 
      if (totalSize() >= 50000000)
        resizeDb(countVals[0].length + 10000000);
      else
        resizeDb((int)(countVals[0].length * 1.6));
    }
  }
  
  public void resizeDb(int newNumKeys) {
	System.out.println("Resizing database to have " + newNumKeys + " keys");
    long[] tempKeys = keys;
    int[][] tempCountVals = countVals;
    keys = new long[newNumKeys];
    countVals = new int[numCountTypes][newNumKeys];
    for (int i = 0; i < tempCountVals[0].length; i++)
    {
      if (tempKeys[i] == 0)
        continue;
      int newDbInd = find(tempKeys[i]);
      keys[newDbInd] = tempKeys[i];
      for (int j = 0; j < numCountTypes; j++)
        countVals[j][newDbInd] = tempCountVals[j][i];
    }
    this.numProbes = 0;
    this.numQueries = 0;
    tempKeys = null;
    tempCountVals = null;
    System.gc();
    System.gc();
    System.gc();
  }
  
  public String getStringAnalysis() {
    int maxBlockSize = 0;
    double avgBlockSize = 0;
    double numBlocks = 0;
    int currSize = 0;
    int[] blockDist = new int[15];
    for (int i = 0; i < totalSize(); i++)
    {
      if (keys[i] != 0)
        currSize++;
      else
      {
        if (currSize > 0)
        {
          numBlocks++;
          avgBlockSize += currSize;
          if (currSize < 15)
            blockDist[currSize]++;
          maxBlockSize = Math.max(maxBlockSize, currSize);
          currSize = 0;
        }
      }
    }
    String retVal = "Total size: " + totalSize()
                   + ", " + numEntries + " entries\n\t" + ((int)numBlocks) + " blocks, avg size "
                   + avgBlockSize/numBlocks + ", max size " + maxBlockSize
                   + "\n\tAverage number of probes: " + ((double)numProbes)/numQueries
                   + "\n\tBlock dist (first few): ";
    for (int i = 0; i < blockDist.length; i++)
      retVal += ((double)blockDist[i])/numBlocks + " ";
    this.numQueries = 0;
    this.numProbes = 0;
    return retVal;
  }
}
