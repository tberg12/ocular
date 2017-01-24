package edu.berkeley.cs.nlp.ocular.eval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import edu.berkeley.cs.nlp.ocular.eval.MarkovEditDistanceComputer.EditDistanceParams;
import tberg.murphy.fileio.f;
import tberg.murphy.tuple.Pair;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class ErrorSampler {
  
  public static class Error implements Comparable<Error> {
    public final int docIdx;
    public final int lineIdx;
    public final int guessTokenIdx;
    public final String guess;
    public final String gold;
    
    public static final String INSERTION = "<INSERTION>";
    public static final String DELETION = "<DELETION>";
    
    public Error(int docIdx, int lineIdx, int guessColumn, String guess, String gold) {
      this.docIdx = docIdx;
      this.lineIdx = lineIdx;
      this.guessTokenIdx = guessColumn;
      this.guess = guess;
      this.gold = gold;
    }

    @Override
    public int compareTo(Error e1) {
      if (this.docIdx != e1.docIdx) {
        return this.docIdx - e1.docIdx;
      } else if (this.lineIdx != e1.lineIdx) {
        return this.lineIdx - e1.lineIdx;
      }
      return this.guessTokenIdx - e1.guessTokenIdx;
    }
    
    public String toString() {
      return "Doc " + docIdx + ", line " + lineIdx + ", guess idx " + guessTokenIdx + ": guess = " + guess + ", gold = " + gold;
    }
    
  }
  
  public static void main(String[] args) {
    List<Error> errors = aggregateWordErrors(args);
    final int NUM_ERRORS = 50;
    Collections.shuffle(errors, new Random(0));
    List<Error> selectedErrors = errors.subList(0, Math.min(errors.size(), NUM_ERRORS));
    Collections.sort(selectedErrors);
    for (int i = 0; i < selectedErrors.size(); i++) {
      System.out.println(selectedErrors.get(i).toString());
    }
  }
  
  public static List<Error> aggregateWordErrors(String[] fileNames) {
    List<Error> allErrors = new ArrayList<Error>();
    for (int fileIdx = 0; fileIdx < fileNames.length; fileIdx++) {
      String fileName = fileNames[fileIdx];
      Pair<List<String>,List<String>> goldGuessLines = getGoldGuessLinesFromOutput(fileName);
      List<String> goldLines = goldGuessLines.getFirst();
      List<String> guessLines = goldGuessLines.getSecond();
      assert goldLines.size() == guessLines.size();
      for (int i = 0; i < goldLines.size(); i++) {
        String goldStr = goldLines.get(i).replaceAll("\\|", "s");
        String guessStr = guessLines.get(i).replaceAll("\\|", "s");
        Form guessForm = Form.wordsAsGlyphs(Arrays.asList(guessStr.split("\\s+")));
        Form goldForm = Form.wordsAsGlyphs(Arrays.asList(goldStr.split("\\s+")));
        EditDistanceParams params = EditDistanceParams.getStandardParams(guessForm, goldForm, false);
        MarkovEditDistanceComputer medc = new MarkovEditDistanceComputer(params);
        AlignedFormPair alignedPair = medc.runEditDistance();
        assert alignedPair.trg.length() == goldForm.length();
        int srcGuessIdx = 0;
        int trgGoldIdx = 0;
        for (Operation op : alignedPair.ops) {
          switch (op) {
            case EQUAL:
              srcGuessIdx++;
              trgGoldIdx++;
              break;
            case SUBST:
              allErrors.add(new Error(fileIdx, i, srcGuessIdx, guessForm.charAt(srcGuessIdx).toString(), goldForm.charAt(trgGoldIdx).toString()));
              srcGuessIdx++;
              trgGoldIdx++;
              break;
            case INSERT:
              allErrors.add(new Error(fileIdx, i, srcGuessIdx, Error.INSERTION, goldForm.charAt(trgGoldIdx).toString()));
              trgGoldIdx++;
              break;
            case DELETE:
              allErrors.add(new Error(fileIdx, i, srcGuessIdx, guessForm.charAt(srcGuessIdx).toString(), Error.DELETION));
              srcGuessIdx++;
              break;
          }
        }
      }
      System.out.println("Processed file " + fileNames[fileIdx] + " with " + goldLines.size() + " lines, cumulative errors = " + allErrors.size());
    }
    return allErrors;
  }
  
  public static Pair<List<String>,List<String>> getGoldGuessLinesFromOutput(String outFile) {
    List<String> lines = f.readLines(outFile);
    List<String> guessLines = new ArrayList<String>();
    List<String> goldLines = new ArrayList<String>();
    for (int i = 0; i < lines.size(); i++) {
      String currLine = lines.get(i).trim();
      if (i % 3 == 0 && currLine.equals("")) {
        break;
      }
      switch (i % 3) {
        case 0: guessLines.add(currLine);
          break;
        case 1: goldLines.add(currLine);
          break;
        case 2: assert currLine.equals("");
          break;
      }
    }
    return Pair.makePair(goldLines, guessLines);
  }
}
