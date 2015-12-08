package edu.berkeley.cs.nlp.ocular.eval;

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of edit distance that supports two nontrivial operations:
 * 
 * a) Switching costs for going from EQUAL to something else or something else to EQUAL.
 * We do not charge for beginning or ending the edit sequence with insertions.
 * 
 * b) Position-specific edit costs for the equality and substitution operations
 * 
 * We need to expand the DP state to do this. Our DP state is actually quite general
 * and could store arbitrary other information; this means that it's somewhat slower
 * than it needs to be because it's object-heavy, but this isn't the bottleneck of our
 * algorithm and this generality allows possible other extensions to edit distance.
 * 
 * @author Greg Durrett (gdurrett@cs.berkeley.edu)
 */
public class MarkovEditDistanceComputer {

  /**
   * Stores parameters for the edit distance operation.
   * Also stores the things being aligned because some parameters are anchored
   * to these, such as the fancy equality and substitute costs.
   */
  public static class EditDistanceParams {

    public final Form src;
    public final Form trg;
    public final double[] equalCosts;
    public final double[] substCosts;
    public final double insertCost;
    public final double deleteCost;
    public final boolean allowFSConfusion;

    public EditDistanceParams(Form src, Form trg, double[] equalCosts, double[] substCosts, double insertCost, double deleteCost, boolean allowFSConfusion) {
      this.src = src;
      this.trg = trg;
      this.equalCosts = equalCosts;
      this.substCosts = substCosts;
      this.insertCost = insertCost;
      this.deleteCost = deleteCost;
      this.allowFSConfusion = allowFSConfusion;
    }

    public static EditDistanceParams getStandardParams(Form src, Form trg, boolean allowFSConfusion) {
      return new EditDistanceParams(src, trg, populateArr(0, src.length()), populateArr(1, src.length()), 1, 1, allowFSConfusion);
    }

    public static double[] populateArr(double val, int len) {
      double[] arr = new double[len];
      for (int i = 0; i < len; i++) {
        arr[i] = val;
      }
      return arr;
    }
  }

  /**
   * State for the Viterbi forward pass through the edit distance lattice to compute backward costs.
   */
  public static class ForwardSearchState {

    public final int srcIndex;
    public final int trgIndex;
    public final double viterbiBackwardCost;
    public final ForwardSearchState viterbiBackptr;

    public ForwardSearchState(int srcIndex, int trgIndex, double viterbiBackwardCost,
                              ForwardSearchState viterbiBackptr) {
      this.srcIndex = srcIndex;
      this.trgIndex = trgIndex;
      this.viterbiBackwardCost = viterbiBackwardCost;
      this.viterbiBackptr = viterbiBackptr;
    }
  }

  private final EditDistanceParams params;
  // Indices are src index, trg index, and previous operations.
  private ForwardSearchState[][] chart;

  public MarkovEditDistanceComputer(EditDistanceParams params) {
    this.params = params;
    this.chart = new ForwardSearchState[params.src.length() + 1][params.trg.length() + 1];
  }

  /**
   * @param op
   * @param state
   * @return The cost to apply the given operator to the given state.
   */
  private double costToApply(Operation op, ForwardSearchState state) {
    if (!isLegalToApply(op, state)) {
      throw new RuntimeException("Illegal operation; applying " + op + " to " + state.srcIndex + ", " + state.trgIndex + " of "
                                 + params.src + "-" + params.trg);
    }
    double cost = 0;
    if (op == Operation.INSERT) {
      cost += params.insertCost;
    } else if (op == Operation.DELETE) {
      cost += params.deleteCost;
    } else if (op == Operation.SUBST) {
      cost += params.substCosts[state.srcIndex];
    } else if (op == Operation.EQUAL) {
      cost += params.equalCosts[state.srcIndex];
    }
    return cost;
  }

  /**
   * @param op
   * @param state
   * @return True if it is legal to apply the given operation to the given state.
   * Checks bounds and conditions for equal vs. substitute
   */
  private boolean isLegalToApply(Operation op, ForwardSearchState state) {
    boolean roomOnSrc = state.srcIndex < params.src.length();
    boolean roomOnTrg = state.trgIndex < params.trg.length();
    if (op == Operation.INSERT) {
      return roomOnTrg;
    } else if (op == Operation.DELETE) {
      return roomOnSrc;
    } else {
      // EQUAL or SUBST must have room on both sides
      if (!roomOnSrc || !roomOnTrg) {
        return false;
      }
      // Now check that EQUAL applies only to equal characters and SUBST only to
      // unequal characters
      Glyph srcGlyph = params.src.charAt(state.srcIndex);
      Glyph trgGlyph = params.trg.charAt(state.trgIndex);
      boolean charsEq = srcGlyph.equals(trgGlyph);
      // Allow permissible confusions with zero cost
      if (params.allowFSConfusion && !charsEq) {
        // Some optimization...
        int srcGlyphLength = srcGlyph.glyph.length();
        int trgGlyphLength = trgGlyph.glyph.length();
        if (srcGlyphLength == trgGlyphLength) {
          if (srcGlyphLength == 1) {
            charsEq = srcGlyph.glyph.equals("f") && trgGlyph.glyph.equals("s");
          } else {
            Glyph newSrc = new Glyph(srcGlyph.glyph.replaceAll("f", "*").replaceAll("s", "*"));
            Glyph newTrg = new Glyph(trgGlyph.glyph.replaceAll("s", "*"));
            charsEq = newSrc.equals(newTrg);
          }
        }
      }
      return (op == Operation.EQUAL && charsEq) || (op == Operation.SUBST && !charsEq);
    }
  }

  /**
   * @param op
   * @param state
   * @return A new state produced by applying op to the given state, or null
   * if op cannot be legally applied here
   */
  private ForwardSearchState apply(Operation op, ForwardSearchState state) {
    if (!isLegalToApply(op, state)) {
      return null;
    }
    int newSrcIndex = state.srcIndex;
    int newTrgIndex = state.trgIndex;
    if (op == Operation.EQUAL || op == Operation.SUBST) {
      newSrcIndex++;
      newTrgIndex++;
    } else if (op == Operation.INSERT) {
      newTrgIndex++;
    } else if (op == Operation.DELETE) {
      newSrcIndex++;
    }
    double costDelta = costToApply(op, state);
    return new ForwardSearchState(newSrcIndex, newTrgIndex, state.viterbiBackwardCost + costDelta, state);
  }

  /**
   * Does the forward pass, computing Viterbi backwards scores for each state.
   */
  private void forwardPass() {
    chart[0][0] = new ForwardSearchState(0, 0, 0, null);
    // Loop over chart cells
    for (int srcIndex = 0; srcIndex < params.src.length() + 1; srcIndex++) {
      if (params.src.length() > 10000 && srcIndex != 0 && srcIndex % 500 == 0) {
        System.out.println("Edit distance working...on srcIndex " + srcIndex + " / " + params.src.length());
      }
      for (int trgIndex = 0; trgIndex < params.trg.length() + 1; trgIndex++) {
        ForwardSearchState prevState = chart[srcIndex][trgIndex];
        if (prevState == null) {
          continue;
        }
        // Loop over operations that could be applied to the given cell
        for (int opIndex = 0; opIndex < Operation.values().length; opIndex++) {
          Operation currOp = Operation.values()[opIndex];
          // Produce the result of applying the operation and insert it into the chart as appropriate
          ForwardSearchState result = apply(currOp, prevState);
          if (result != null) {
            ForwardSearchState currEntry = chart[result.srcIndex][result.trgIndex];
            if (currEntry == null || result.viterbiBackwardCost < currEntry.viterbiBackwardCost) {
              chart[result.srcIndex][result.trgIndex] = result;
            }
          }
        }
      }
    }
  }

  /**
   * Moves back through the chart and extracts the one-best solution.
   * @return The forms being aligned here and their one-best alignment.
   */
  private AlignedFormPair backwardPass() {
    ForwardSearchState currState = chart[params.src.length()][params.trg.length()];
    if (currState == null) {
      throw new RuntimeException("Edit distance returned nothing for " + params.src + "-" + params.trg);
    }
    double cost = currState.viterbiBackwardCost;
    List<Operation> ops = new ArrayList<Operation>();
    // Until we hit the first state, accrue the edit ops (which come in reverse order)
    while (currState.viterbiBackptr != null) {
      // Figure out which operation was used
      int thisSrcIdx = currState.srcIndex;
      int thisTrgIdx = currState.trgIndex;
      int prevSrcIdx = currState.viterbiBackptr.srcIndex;
      int prevTrgIdx = currState.viterbiBackptr.trgIndex;
      Operation op;
      if (prevSrcIdx == thisSrcIdx) {
        op = Operation.INSERT;
      } else if (prevTrgIdx == thisTrgIdx) {
        op = Operation.DELETE;
      } else {
        if (params.src.charAt(prevSrcIdx).equals(params.trg.charAt(prevTrgIdx))) {
          op = Operation.EQUAL;
        } else {
          op = Operation.SUBST;
        }
      }
      ops.add(0, op);
      currState = currState.viterbiBackptr;
    }
    return new AlignedFormPair(params.src, params.trg, ops, cost);
  }

  public AlignedFormPair runEditDistance() {
    if (params.src.length() > 10000) {
      System.out.println("Running edit distance with source length " + params.src.length() + ", for src length 7000 takes 30 seconds and 1+GB of memory");
    }
    forwardPass();
    return backwardPass();
  }

}
