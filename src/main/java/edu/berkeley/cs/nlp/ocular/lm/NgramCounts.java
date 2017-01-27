package edu.berkeley.cs.nlp.ocular.lm;

import java.util.Arrays;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class NgramCounts {
  public final NgramWrapper ngram;
  public final CountDbBig[] counts;
  
  public final long[] tokenCounts;
  public final long[] tokenNormalizers;
  public final long[] typeCounts;
  public final long[] typeNormalizers;
  public final long[] historyTypeCounts;
  
  public static final double UNK_LOG_PROB = -10;
  public static final double DISCOUNT = 0.75;
  
  public NgramCounts(NgramWrapper ngram, CountDbBig[] counts) {
//    Logger.logss("ORIGINAL NGRAM: " + Ngram.toString(ngram, charIndexer));
    this.ngram = ngram;
    this.counts = counts;
    int ngramOrder = ngram.getOrder();
    this.tokenCounts = new long[ngramOrder];
    this.tokenNormalizers = new long[ngramOrder];
    this.historyTypeCounts = new long[ngramOrder];
    int numberOfTypeCounts = Math.min(ngramOrder, counts.length-1);
    this.typeCounts = new long[numberOfTypeCounts];
    this.typeNormalizers = new long[numberOfTypeCounts];
    for (int i = 0; i < ngramOrder; i++) {
      int order = i + 1;
      // Version of the current n-gram, truncated to the appropriate length
      NgramWrapper tempNgramWrapper = ngram.getLowerOrder(order);
      long[] tempNgram = tempNgramWrapper.getLongerRep();
      long[] tempHistory = tempNgramWrapper.getHistory().getLongerRep();
      this.tokenCounts[i] = counts[i].getCount(tempNgram, CountType.TOKEN_INDEX);
      if (i > 0) {
        this.tokenNormalizers[i] = counts[i-1].getCount(tempHistory, CountType.TOKEN_INDEX);
        this.historyTypeCounts[i] = counts[i-1].getCount(tempHistory, CountType.HISTORY_TYPE_INDEX);
      } else {
        this.tokenNormalizers[i] = counts[i].getNumTokens();
        this.historyTypeCounts[i] = 0; // don't need these at the lowest order
      }
      if (i < numberOfTypeCounts) {
        this.typeCounts[i] = counts[i].getCount(tempNgram, CountType.LOWER_ORDER_TYPE_INDEX);
        if (i > 0) {
          this.typeNormalizers[i] = counts[i-1].getCount(tempHistory, CountType.LOWER_ORDER_TYPE_NORMALIZER);
        } else {
          this.typeNormalizers[i] = counts[0].getNumBigramTypes();
        }
      }
    }
  }
  
  public int getNgramOrder() {
    return ngram.getOrder();
  }
  
  /**
   * @return The highest order for which we have nonzero history counts
   */
  public int getHighestUsableOrder() {
    for (int i = getNgramOrder() - 1; i >= 0; i--) {
      if (tokenCounts[i] > 0) {
        //if (tokenNormalizers[i] <= 0) throw new RuntimeException("Bad counts: " + this);
      }
      if (tokenNormalizers[i] > 0) {
        return i+1;
      }
    }
    throw new RuntimeException("getHighestUsableOrder() failed.  getNgramOrder()="+getNgramOrder());
  }
  
  public double getTokenMle() {
    return getTokenMle(getHighestUsableOrder() - 1);
  }
  
  public double getTokenMle(int orderIndex) {
    return ((double)tokenCounts[orderIndex])/((double)tokenNormalizers[orderIndex]);
  }
  
  public double getTokenMleOrEpsilon(int orderIndex) {
    if (tokenCounts[orderIndex] == 0) {
      return Math.exp(UNK_LOG_PROB);
    } else {
      return ((double)tokenCounts[orderIndex])/((double)tokenNormalizers[orderIndex]);
    }
  }
  
  public double getTypeMle(int orderIndex) {
    return ((double)typeCounts[orderIndex])/((double)typeNormalizers[orderIndex]);
  }
  
  public double getAbsoluteDiscounting() {
    return adHelper(getHighestUsableOrder());
  }
  
  private double adHelper(int order) {
//    Logger.logss("AD ORDER: " + order);
    int orderIndex = order - 1;
    if (order == 1) {
      return getTokenMleOrEpsilon(orderIndex);
    } else {
      return (Math.max(0.0, ((double)tokenCounts[orderIndex]) - DISCOUNT))/((double)tokenNormalizers[orderIndex])
          + ((double)historyTypeCounts[orderIndex]) * DISCOUNT/((double)tokenNormalizers[orderIndex])
          * adHelper(order - 1);
    }
  }
  
  public double getKneserNey() {
    int highestOrder = getHighestUsableOrder();
    int highestOrderIndex = highestOrder - 1;
    if (highestOrder == 1) {
      return getTokenMleOrEpsilon(highestOrderIndex);
    } else if (highestOrder == getNgramOrder()) {
      double alpha = (Math.max(0.0, ((double)tokenCounts[highestOrderIndex]) - DISCOUNT))/((double)tokenNormalizers[highestOrderIndex]);
      double bow = ((double)historyTypeCounts[highestOrderIndex]) * DISCOUNT/((double)tokenNormalizers[highestOrderIndex]);
//      Logger.logss("KNTOP: " + alpha + "   " + bow);
      return alpha + bow * knHelper(highestOrder - 1);
    } else {
      return knHelper(highestOrder);
    }
  }
  
  private double knHelper(int order) {
    int orderIndex = order - 1;
    if (order == 1) {
      if (typeCounts[0] == 0) {
        return Math.exp(UNK_LOG_PROB);
      } else {
        return ((double)typeCounts[0])/((double)typeNormalizers[0]);
      }
    } else {
      double alpha = (Math.max(0.0, ((double)typeCounts[orderIndex]) - DISCOUNT))/((double)typeNormalizers[orderIndex]);
      double bow = ((double)historyTypeCounts[orderIndex]) * DISCOUNT/((double)typeNormalizers[orderIndex]);
//      Logger.logss("KN: " + alpha + "   " + bow);
      return alpha + bow * knHelper(order - 1);
    }
  }
  
  public String toString() {
    String string = "";
    string += "Ngram: " + ngram + "; order: " + getNgramOrder() + "\n";
    string += "Tok: " + Arrays.toString(tokenCounts) + "\n";
    string += "TokNorm: " + Arrays.toString(tokenNormalizers) + "\n";
    string += "Typ: " + Arrays.toString(typeCounts) + "\n";
    string += "TypNorm: " + Arrays.toString(typeNormalizers) + "\n";
    string += "HistTyp: " + Arrays.toString(historyTypeCounts) + "\n";
    return string;
  }
}
