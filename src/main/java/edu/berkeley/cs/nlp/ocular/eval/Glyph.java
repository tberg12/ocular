package edu.berkeley.cs.nlp.ocular.eval;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class Glyph implements Comparable<Glyph> {
  
  public final String glyph;
  
  public Glyph(String glyph) {
    this.glyph = glyph;
  }

  @Override
  public boolean equals(Object other) {
    if (other == null || !(other instanceof Glyph)) {
      return false;
    }
    return this.glyph.equals(((Glyph)other).glyph);
  }
  
  @Override
  public int hashCode() {
    return glyph.hashCode();
  }

  @Override
  public String toString() {
    return glyph;
  }

  @Override
  public int compareTo(Glyph o) {
    return this.glyph.compareTo(o.glyph);
  }
}
