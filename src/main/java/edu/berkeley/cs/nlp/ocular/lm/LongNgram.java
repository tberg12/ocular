package edu.berkeley.cs.nlp.ocular.lm;

import tberg.murphy.indexer.Indexer;

/**
 * Contains code for carrying out operations on trigrams encoded as longs.
 * Can be instantiated, but also has static methods so that the code can be
 * used without creating the object.
 * 
 * Indices are packed into a long using BITS_PER_WORD bits per index,
 * up to MAX_ORDER indices. BITS_PER_WORD * MAX_ORDER must be <= 64 (use = at your own risk...)
 * When indices are in the long, 1 is added to each of them so that lower-order
 * n-grams (with 0s) can be differentiated from n-grams with the first character in the indexer
 * in them.
 * 
 * @author Greg Durrett (gdurrett@cs.berkeley.edu)
 */
public class LongNgram {

  // 128 characters should be enough, this lets us do a 9-gram
  public static final int BITS_PER_WORD = 7;
  public static final int MAX_ORDER = 9;

  public static long[] convertToLong(int[] ngram) {
    return convertToLong(ngram, 0, ngram.length);
  }

  public static long[] convertToLong(int[] ngram, int start, int end) {
    // Add MAX_ORDER-1 to round up
    int numLongs = (end - start + MAX_ORDER-1)/MAX_ORDER;
    long[] longs = new long[numLongs];
    int longIdx = numLongs - 1;
    for (int i = end; i > start; i -= MAX_ORDER) {
      longs[longIdx] = Ngram.convertToLong(ngram, Math.max(start, i - MAX_ORDER), i);
      longIdx--;
    }
    return longs;
  }

  public static int[] convertToIntArr(long[] ngram) {
    int[] arr = new int[LongNgram.getActualOrder(ngram)];
    int ngramIdx = arr.length - 1;
    for (int longIdx = ngram.length - 1; longIdx >= 0; longIdx--) {
      int[] curr = Ngram.convertToIntArr(ngram[longIdx]);
      for (int i = curr.length - 1; i >= 0; i--) {
        arr[ngramIdx] = curr[i];
        ngramIdx--;
      }
    }
    return arr;
  }

  // TODO: I think these methods work but they don't do clipping to arbitrary orders,
  // and I think it's easier to just 
//  public static long[] getLowerOrder(long[] ngram) {
//    return LongNgram.getLowerOrder(ngram, LongNgram.getActualOrder(ngram));
//  }
//
//  public static long[] getLowerOrder(long[] ngram, int order) {
//    if (order % MAX_ORDER == 1) {
//      long[] newNgram = new long[ngram.length-1];
//      System.arraycopy(ngram, 1, newNgram, 0, ngram.length-1);
//      return newNgram;
//    } else {
//      long[] newNgram = new long[ngram.length];
//      System.arraycopy(ngram, 0, newNgram, 0, ngram.length);
//      newNgram[0] = Ngram.getLowerOrder(ngram[0]);
//      return newNgram;
//    }
//  }
//
//  public static long[] getHistory(long[] ngram) {
//    return LongNgram.getHistory(ngram, LongNgram.getActualOrder(ngram));
//  }
//
//  public static long[] getHistory(long[] ngram, int order) {
//    long lowOrderMask = (1L << ((long)BITS_PER_WORD)) - 1L;
//    long[] newNgram;
//    int newNgramIdx;
//    long carryOver;
//    if (order % MAX_ORDER == 1) {
//      newNgram = new long[ngram.length-1];
//      newNgramIdx = 0;
//      carryOver = ngram[0];
//    } else {
//      newNgram = new long[ngram.length];
//      newNgramIdx = 1;
//      carryOver = ngram[0] & lowOrderMask;
//      newNgram[0] = ngram[0] >>> BITS_PER_WORD;
//    }
//    for (int i = 1; i < ngram.length; i++) {
//      newNgram[newNgramIdx] = ngram[i] >>> BITS_PER_WORD + carryOver << (BITS_PER_WORD * (MAX_ORDER - 1));
//      newNgramIdx++;
//      carryOver = ngram[i] & lowOrderMask;
//    }
//    return newNgram;
//  }
//
//  public static long[] getLowerOrderHistory(long[] ngram) {
//    return LongNgram.getLowerOrderHistory(ngram, LongNgram.getActualOrder(ngram));
//  }
//
//  public static long[] getLowerOrderHistory(long[] ngram, int order) {
//    return LongNgram.getLowerOrder(LongNgram.getHistory(ngram, order), order - 1);
//  }

  public static int getActualOrder(long[] ngram) {
    if (ngram.length == 0) {
      return 0;
    } else {
      return (ngram.length - 1) * MAX_ORDER + Ngram.getActualOrder(ngram[0]);
    }
  }

  public static String toString(int[] ngram, Indexer<String> indexer) {
    return LongNgram.toString(LongNgram.convertToLong(ngram), indexer);
  }

  public static String toString(long[] ngram, Indexer<String> indexer) {
    int order = LongNgram.getActualOrder(ngram);
    String ngramStr = "";
    for (int i = 0; i < ngram.length; i++) {
      ngramStr += Ngram.getNgramStr(ngram[i], indexer);
    }
    return "[" + order + ":" + ngramStr + "]";
  }
}
