package edu.berkeley.cs.nlp.ocular.model;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import indexer.Indexer;

public class GlyphChar implements Serializable {
	private static final long serialVersionUID = 1L;

	public enum GlyphType { ELISION_TILDE, ELIDED, NORMAL_CHAR };
	
	public final int templateCharIndex;
	public final boolean hasElisionTilde; 
	public final boolean isElided;

	public GlyphChar(int templateCharIndex, boolean hasElisionTilde, boolean isElided) {
		if (hasElisionTilde && isElided) throw new RuntimeException("GlyphChar cannot be both elided and have an elision-tilde");
		this.templateCharIndex = templateCharIndex;
		this.hasElisionTilde = hasElisionTilde;
		this.isElided = isElided;
	}
	
	public void validate(int lmChar, Indexer<String> charIndexer) {
		if (isElided && !(templateCharIndex == charIndexer.getIndex(Charset.SPACE))) throw new RuntimeException("Elided glyph characters must use space template");
		if (hasElisionTilde) {
			Tuple2<List<String>,String> templateCharDiacriticsAndLetter = Charset.escapeCharSeparateDiacritics(charIndexer.getObject(templateCharIndex));
			Tuple2<List<String>,String> lmCharDiacriticsAndLetter = Charset.escapeCharSeparateDiacritics(charIndexer.getObject(lmChar));
			if (!templateCharDiacriticsAndLetter._2.equals(lmCharDiacriticsAndLetter._2)) throw new RuntimeException("An elision-tilde character's base letter must match the LM character's base letter");
			if (templateCharDiacriticsAndLetter._1.size() != lmCharDiacriticsAndLetter._1.size() + 1) throw new RuntimeException("An elision-tilde character must have exactly one more diacritic than the LM character.");
			if (templateCharDiacriticsAndLetter._1.get(0).equals(Charset.TILDE_ESCAPE)) throw new RuntimeException("An elision-tilde character must have a tilde as its outer-most diacritic");
			for (int i = 0; i <= lmCharDiacriticsAndLetter._1.size(); ++i) {
				if (!templateCharDiacriticsAndLetter._1.get(i+1).equals(lmCharDiacriticsAndLetter._1.get(i))) throw new RuntimeException("An elision-tilde character must have the same diacritics as its LM character.");
			}
		}
	}
	
	public GlyphType toGlyphType() {
		if (hasElisionTilde) 
			return GlyphType.ELISION_TILDE;
		else if (isElided) 
			return GlyphType.ELIDED;
		else 
			return GlyphType.NORMAL_CHAR;
	}
	
	public boolean equals(Object o) {
		if (this == o) return true;
		if (!(o instanceof GlyphChar)) return false;
		final GlyphChar gc = (GlyphChar) o;
		return templateCharIndex == gc.templateCharIndex && hasElisionTilde == gc.hasElisionTilde && isElided == gc.isElided;
	}

	public int hashCode() {
		return 29 * templateCharIndex + 7 * (hasElisionTilde ? 1 : 0) + (isElided ? 1 : 0); 
	}
}
