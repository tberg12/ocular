package edu.berkeley.cs.nlp.ocular.gsm;

import java.io.Serializable;

import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class GlyphChar implements Serializable {
	private static final long serialVersionUID = 1L;

	public enum GlyphType { 
		ELISION_TILDE, //                 this glyph is marked with a tilde indicating that some subsequent letter have been elided   
		TILDE_ELIDED, //                  this (empty) glyph appears after an "elision tilde"
		FIRST_ELIDED, //                  this (empty) glyph results from the elision of the first letter of a word  
		DOUBLED, //                       this glyph marks an empty LM character whose glyph is a duplicate of the next glyph, which is just a rendering of its LM character
		//RMRGN_HPHN_DROP, //               this glyph marks a right-margin line-breaking hyphen is not printed 
		ELIDED, //                        this (empty) glyph results from the elision a character  
		NORMAL_CHAR }; //                 
	
	public final int templateCharIndex;
	public final GlyphType glyphType;

	public GlyphChar(int templateCharIndex, GlyphType glyphType) {
		this.templateCharIndex = templateCharIndex;
		this.glyphType = glyphType;
	}
	
	public boolean isElided() {
		switch (glyphType) {
			case TILDE_ELIDED:
			case FIRST_ELIDED:
			case ELIDED:
				return true;
			default:
				return false;
		}
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
		return "GlyphChar("+charIndexer.getObject(templateCharIndex)+", "+glyphType+")";
	}
}
