package edu.berkeley.cs.nlp.ocular.eval;

import java.util.List;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class AlignedFormPair {
  
  public final Form src;
  public final Form trg;
  public final List<Operation> ops;
  public final double cost;
  
  public AlignedFormPair(Form src, Form trg, List<Operation> ops, double cost) {
    this.src = src;
    this.trg = trg;
    this.ops = ops;
    this.cost = cost;
  }
}
