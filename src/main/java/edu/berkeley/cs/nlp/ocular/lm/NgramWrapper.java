package edu.berkeley.cs.nlp.ocular.lm;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class NgramWrapper {

  public int[] ngram;
  public int start;
  public int end;

  private NgramWrapper() {
    this.ngram = null;
    this.start = -1;
    this.end = -1;
  }

  public static NgramWrapper getNew(int[] ngram, int start, int end) {
    NgramWrapper next = new NgramWrapper();
    next.changeNgramWrapper(ngram, start, end);
    return next;
  }

  private void changeNgramWrapper(int[] ngram, int start, int end) {
    this.ngram = ngram;
    this.start = start;
    this.end = end;
  }

  public int getOrder() {
    return end - start;
  }

  public NgramWrapper getLowerOrder() {
    return getNew(ngram, start + 1, end);
  }

  public NgramWrapper getLowerOrder(int order) {
    return getNew(ngram, end - order, end);
  }

  public NgramWrapper getHistory() {
    return getNew(ngram, start, end - 1);
  }

  public long getLongRep() {
    return Ngram.convertToLong(ngram, start, end);
  }

  public long[] getLongerRep() {
    return LongNgram.convertToLong(ngram, start, end);
  }
  
  public String toString() {
    String str = "[";
    for (int i = start; i < end; i++) {
      str += ngram[i] + ", ";
    }
    if (str.length() == 1) {
      return str + "]";
    } else {
      return str.substring(0, str.length() - 2) + "]";
    }
  }
}
