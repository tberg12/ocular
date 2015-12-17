package edu.berkeley.cs.nlp.ocular.sub;

import java.io.Serializable;

import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class GlyphChar implements Serializable {
	private static final long serialVersionUID = 1L;

	public enum GlyphType { ELISION_TILDE, TILDE_ELIDED, FIRST_ELIDED, NORMAL_CHAR };
	
	public final int templateCharIndex;
	public final GlyphType glyphType;

	public GlyphChar(int templateCharIndex, GlyphType glyphType) {
		this.templateCharIndex = templateCharIndex;
		this.glyphType = glyphType;
	}
	
	public boolean isElided() {
		return glyphType == GlyphType.TILDE_ELIDED || glyphType == GlyphType.FIRST_ELIDED;
	}
	
	public GlyphType toGlyphType() {
		return glyphType;
	}
	
	public boolean equals(Object o) {
		if (this == o) return true;
		if (!(o instanceof GlyphChar)) return false;
		final GlyphChar gc = (GlyphChar) o;
		return templateCharIndex == gc.templateCharIndex && glyphType == gc.glyphType;
	}

	public int hashCode() {
		return 29 * templateCharIndex + 17 * (glyphType.ordinal()); 
	}
	
	public String toString() {
		return "GlyphChar(templateCharIndex="+templateCharIndex+", glyphType="+glyphType+")";
	}
	
	public String toString(Indexer<String> charIndexer) {
		return "GlyphChar(tmplChar="+charIndexer.getObject(templateCharIndex)+"("+templateCharIndex+"), glyphType="+glyphType+")";
	}
}
