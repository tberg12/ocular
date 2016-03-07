package edu.berkeley.cs.nlp.ocular.lm;

import indexer.Indexer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class CorpusCounter {

  public final CountDbBig[] counts;
  public final int maxNgramOrder;
  private long tokenCount = 0;
  // If you set the size of each hash table to be at least, say, 1.4 * the number of elements
  // it will have to contain, you'll never have to resize while constructing the LM and you'll
  // end up with good performance. It will auto-resize if you need more but this could potentially
  // make some of the tables blow up too much.
  public final int MILLION = 1000000;
  // XXX: Uncomment this one if you want to run on less data and use less memory
//  public final int[] INITIAL_CHAR_DB_SIZES = { 100, 1000, 3000, 9000, 20000, 60000, 120000, 400000, 1000000 };
  public final int[] INITIAL_CHAR_DB_SIZES = { 100, 6000, 60000, 300000, 1 * MILLION, 3 * MILLION, 6 * MILLION, 10 * MILLION, 20 * MILLION,
                                               40 * MILLION, 60 * MILLION, 80 * MILLION };

  private final Set<Integer> activeCharacters;
  private final Map<Integer,Integer> unigramCounts;
  
	public CorpusCounter(int maxNgramOrder) {
    this.counts = new CountDbBig[maxNgramOrder];
    int[] dbSizes = new int[maxNgramOrder];
    for (int i = 0; i < dbSizes.length; i++) {
      if (i < INITIAL_CHAR_DB_SIZES.length) {
        dbSizes[i] = INITIAL_CHAR_DB_SIZES[i];
      } else {
        dbSizes[i] = 100 * MILLION;
      }
    }
    for (int i = 0; i < maxNgramOrder - 2; i++) {
      this.counts[i] = new CountDbBig(dbSizes[i], 4);
    }
    this.counts[maxNgramOrder - 2] = new CountDbBig(dbSizes[maxNgramOrder - 2], 3);
    this.counts[maxNgramOrder - 1] = new CountDbBig(dbSizes[maxNgramOrder - 1], 1);
    this.maxNgramOrder = maxNgramOrder;
    this.tokenCount = 0;
    
    this.activeCharacters = new TreeSet<Integer>();
    this.unigramCounts = new HashMap<Integer,Integer>();
  }

  public CountDbBig[] getCounts() {
    return counts;
  }
  
  /**
   * Count either a file or all the files/subdirectories recursively in a directory
   */
  public void countRecursive(String name, int maxNumLines, Indexer<String> charIndexer, TextReader textReader) {
	  System.out.println("CorpusCounter:  Count recursive starting from "+name);
	  for(File file : FileUtil.recursiveFiles(name)) {
		  System.out.println("    counting file: "+file);
		  count(file.getPath(), maxNumLines, charIndexer, textReader);
	  }
  }

  public void count(String fileName, int maxNumLines, Indexer<String> charIndexer, TextReader textReader) {
    try {
      BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
      int lineNumber = 0;
      while (in.ready()) {
        if (lineNumber >= maxNumLines) {
          break;
        }
        String line = in.readLine();
        countLine(line, charIndexer, textReader, lineNumber);
        lineNumber++;
      }
      in.close();
      //printStats(lineNumber);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void countLine(String line, Indexer<String> charIndexer, TextReader textReader, int lineNumber) {
      List<String> chars = textReader.readCharacters(line);
      countChars(chars, charIndexer, lineNumber);
  }

  public void countChars(List<String> chars, Indexer<String> charIndexer, int lineNumber) {
      int[] indexedLine = new int[chars.size()];
      int t = 0;
      for (String currChar : chars) {
        if (charIndexer.locked() && !charIndexer.contains(currChar)) {
          // TODO: Change if we want to use UNK instead of -1
          indexedLine[t++] = -1; // charIndexer.getIndex("UNK");
  	    //throw new AssertionError("Found unknown character `"+currChar+"` in file `"+fileName+"`");
        } else {
          indexedLine[t++] = charIndexer.getIndex(currChar);
        }
      }
      countLine(indexedLine, lineNumber);
  }

  public void count(int[][] text) {
    for (int i = 0; i < text.length; i++) {
      countLine(text[i], i);
    }
    //printStats(text.length);
  }
  
  /**
   * Line may contain -1s, which are essentially treated as newlines
   * @param line
   * @param lineIdx
   */
  public void countLine(int[] line, int lineIdx) {
    // Put in two start of sentence tokens
    int[] ngramArr = new int[maxNgramOrder];
    Arrays.fill(ngramArr, -1);
    for (int charIdx = 0; charIdx < line.length; charIdx++) {
      for (int i = 0; i < ngramArr.length - 1; i++) {
        ngramArr[i] = ngramArr[i+1];
      }
      ngramArr[ngramArr.length - 1] = line[charIdx];
      // If the first -1 isn't present, then we get maxNgramOrder, otherwise it
      // drops by one for every eaten position; this is correct.
      if (line[charIdx] != -1) {
        incrementCounts(ngramArr, maxNgramOrder - (firstMinusOneLookingBack(ngramArr) + 1));
        
        int c = line[charIdx];
        this.activeCharacters.add(c);
        this.unigramCounts.put(c, CollectionHelper.getOrElse(this.unigramCounts, c, 0) + 1);
      }
      tokenCount++;
      for (int i = 0; i < counts.length; i++) {
        counts[i].maybeResize();
      }
    }
  }
  
  private int firstMinusOneLookingBack(int[] arr) {
    int i;
    for (i = arr.length - 1; i >= 0; i--) {
      if (arr[i] == -1) {
        return i;
      }
    }
    return -1;
  }
  
  public void printStats(int lineIdx) {
    System.out.println("=============================================");
    System.out.println("Line " + lineIdx);
    System.out.println("Number of tokens: train: " + tokenCount);
    for (int i = 0; i < counts.length; i++) {
    	System.out.println((i+1) + "-gram DB:\n\t" + counts[i].getStringAnalysis());
    	System.out.println("\t" + (i+1) + "-grams total and curr: " + counts[i].totalSize() + " " + counts[i].currSize());
    }
  }

  private void incrementCounts(int[] ngramArr, int order) {
    assert order >= 1;
    // Increment token and type counts for the highest-order n-gram
    // Lower-order token counts that involve this order of n-gram are stored
    // and indexed in lower-order count databases, but incremented here
    NgramWrapper ngram = NgramWrapper.getNew(ngramArr, ngramArr.length - order, ngramArr.length);
//    System.out.println("ngram " + ngram.toString() + ", back " + Arrays.toString(LongNgram.convertToIntArr(ngram.getLongerRep())));
    int oldTokenCount = counts[order - 1].incrementCount(ngram, CountType.TOKEN_INDEX);
    if (oldTokenCount == 0 && order > 1) {
      NgramWrapper lowerOrder = ngram.getLowerOrder();
      NgramWrapper history = ngram.getHistory();
//      System.out.println("lo " + Arrays.toString(LongNgram.convertToIntArr(lowerOrder.getLongerRep())));
//      System.out.println("hist " + Arrays.toString(LongNgram.convertToIntArr(history.getLongerRep())));
      counts[order - 2].incrementCount(lowerOrder, CountType.LOWER_ORDER_TYPE_INDEX);
      counts[order - 2].incrementCount(history, CountType.HISTORY_TYPE_INDEX);
      if (order > 2) {
        NgramWrapper lowerOrderHistory = history.getLowerOrder();
        counts[order - 3].incrementCount(lowerOrderHistory, CountType.LOWER_ORDER_TYPE_NORMALIZER);
      } else { // order == 2
        counts[order - 2].incrementBigramTypes();
      }
    }
    // Recurse to count lower-order stuff
    if (order > 1) {
      incrementCounts(ngramArr, order - 1);
    }
  }
  
  public long getTokenCount() {
	  return tokenCount;
  }
  
  public Set<Integer> getActiveCharacters() {
		return activeCharacters;
  }
  
  public Map<Integer,Integer> getUnigramCounts() {
		return unigramCounts;
  }

}
