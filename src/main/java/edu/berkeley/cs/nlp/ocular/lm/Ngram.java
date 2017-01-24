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
public class Ngram {

  // 128 characters should be enough, this lets us do a 9-gram
  public static final int BITS_PER_WORD = 7;
  public static final int MAX_ORDER = 9;
  public static final int[] CONVERTER = new int[MAX_ORDER];

  private static int encodeWord(int rawWord) {
    return rawWord + 1;
  }

  private static int decodeWord(int encodedWord) {
    return encodedWord - 1;
  }

  public static long convertToLong(int[] ngram) {
    return convertToLong(ngram, 0, ngram.length);
  }

  public static long convertToLong(int[] ngram, int start, int end) {
    long l = 0;
    for (int i = start; i < end; i++)
      l = (l << BITS_PER_WORD) + encodeWord(ngram[i]);
    return l;
  }

  public static int[] convertToIntArr(long ngram) {
    //    assert Ngram.getActualOrder(ngram) == MAX_ORDER : "Ngram of less than max order: "
    //              + Ngram.toString(ngram) + ", order: " + Ngram.getActualOrder(ngram);
    int[] arr = new int[Ngram.getActualOrder(ngram)];
    int i = 0;
    long wordMask = (1L << BITS_PER_WORD) - 1;
    while (ngram != 0) {
      arr[arr.length - 1 - i] = decodeWord((int) (ngram & wordMask));
      i++;
      ngram = Ngram.getHistory(ngram);
    }
    return arr;
  }

  public static long getLowerOrder(long ngram) {
    return Ngram.getLowerOrder(ngram, Ngram.getActualOrder(ngram));
  }

  public static long getLowerOrder(long ngram, int order) {
    long mask = (1L << ((order - 1) * BITS_PER_WORD)) - 1L;
    return mask & ngram;
  }

  public static long getHistory(long ngram) {
    return Ngram.getHistory(ngram, Ngram.getActualOrder(ngram));
  }

  public static long getHistory(long ngram, int order) {
    long mask = ((1L << (((long) order - 1) * BITS_PER_WORD)) - 1L) << BITS_PER_WORD;
    return (mask & ngram) >> BITS_PER_WORD;
  }

  public static long getLowerOrderHistory(long ngram) {
    return Ngram.getLowerOrderHistory(ngram, Ngram.getActualOrder(ngram));
  }

  public static long getLowerOrderHistory(long ngram, int order) {
    return Ngram.getLowerOrder(Ngram.getHistory(ngram, order), order - 1);
  }

//  public static long addWordAndShift(long ngram, int word) {
//    long mask = (1L << (((long) MAX_ORDER - 1) * BITS_PER_WORD)) - 1L << BITS_PER_WORD;
//    return ((ngram << BITS_PER_WORD) & mask) + encodeWord(word);
//  }

  public static int getActualOrder(long ngram) {
    for (int i = MAX_ORDER - 1; i >= 0; i--) {
      long mask = (1L << (((long) i) * BITS_PER_WORD)) - 1L;
      if ((ngram & mask) != ngram)
        return i + 1;
    }
    return 0;
  }

  public static String toString(int[] ngram, Indexer<String> indexer) {
    return Ngram.toString(Ngram.convertToLong(ngram), indexer);
  }

  public static String toString(long ngram, Indexer<String> indexer) {
    return "[" + Ngram.getActualOrder(ngram) + ":" + getNgramStr(ngram, indexer) + "]";
  }
  
  public static String getNgramStr(long ngram, Indexer<String> indexer) {
    String string = "";
    int order = Ngram.getActualOrder(ngram);
    for (int i = 0; i < order; i++) {
      long mask = (1L << BITS_PER_WORD) - 1L;
      string = indexer.getObject(decodeWord((int) (ngram & mask))) + string;
      ngram = ngram >> BITS_PER_WORD;
    }
    return string;
  }
}
