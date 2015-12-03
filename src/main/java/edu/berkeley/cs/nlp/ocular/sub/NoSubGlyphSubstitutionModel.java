package edu.berkeley.cs.nlp.ocular.sub;

import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class NoSubGlyphSubstitutionModel implements GlyphSubstitutionModel {
	private static final long serialVersionUID = 1L;

	private Indexer<String> langIndexer;
	private Indexer<String> charIndexer;

	public NoSubGlyphSubstitutionModel(Indexer<String> langIndexer, Indexer<String> charIndexer) {
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
	}

	public Indexer<String> getLanguageIndexer() {
		return langIndexer;
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}
	
	public double logGlyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		return ((!glyphChar.hasElisionTilde && !glyphChar.isElided && lmChar == glyphChar.templateCharIndex) ? 0.0 : Double.NEGATIVE_INFINITY);
	}
	
}
