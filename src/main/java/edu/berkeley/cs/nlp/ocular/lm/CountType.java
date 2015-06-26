package edu.berkeley.cs.nlp.ocular.lm;

public enum CountType
{
  TOKEN_INDEX(0),
  HISTORY_TYPE_INDEX(1),
  LOWER_ORDER_TYPE_INDEX(2),
  LOWER_ORDER_TYPE_NORMALIZER(3);
  
  private final int index;
  
  private CountType(int index) {
    this.index = index;
  }
  
  public int getIndex() {
    return index;
  }
}