package edu.berkeley.cs.nlp.ocular.eval;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public enum Operation {

  EQUAL, SUBST, INSERT, DELETE;
  
  public static String opToString(Operation op) {
    switch (op) {
      case EQUAL: return "=";
      case SUBST: return "S";
      case INSERT : return "I";
      case DELETE : return "D";
      default : throw new RuntimeException("Bad op: " + op);
    }
  }

  public static String opsToString(List<Operation> ops) {
    String opsStr = "";
    for (Operation op : ops) {
      opsStr += opToString(op);
    }
    return opsStr;
  }
  
  public static Operation charToOp(char opChar) {
    switch (opChar) {
      case '=': return EQUAL;
      case 'S': return SUBST;
      case 'I': return INSERT;
      case 'D': return DELETE;
      default : throw new RuntimeException("Bad op string: " + opChar);
    }
  }

  public static List<Operation> stringToOps(String opsStr) {
    List<Operation> ops = new ArrayList<Operation>();
    for (int i = 0; i < opsStr.length(); i++) {
      ops.add(charToOp(opsStr.charAt(i)));
    }
    return ops;
  }
}
