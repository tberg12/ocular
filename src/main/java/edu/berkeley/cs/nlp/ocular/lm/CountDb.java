package edu.berkeley.cs.nlp.ocular.lm;


public interface CountDb {
  
  public long getNumTokens();
  
  public int getNumBigramTypes();
  
  public int currSize();
  
  public int totalSize();
  
  public long[] getKeys();

  public int getCount(long key, CountType countType);
  
  public int getCount(NgramWrapper ngram, CountType countType);
  
  public void incrementBigramTypes();
  
  /**
   * @return The old count of the ngram (pre-update), but only if we do token counts
   */
  public int incrementCount(NgramWrapper ngram, CountType countType);
  
  public void maybeResize();
  
  public String getStringAnalysis();
}
