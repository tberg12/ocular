package edu.berkeley.cs.nlp.ocular.gsm;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.TILDE_COMBINING;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import tberg.murphy.indexer.HashMapIndexer;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicGlyphSubstitutionModelTests {

	@Test
	public void test_getSmoothingValue() {

		double gsmSmoothingCount = 0.1;
		double gsmElisionSmoothingCountMultiplier = 500.0;
		Indexer<String> langIndexer = new HashMapIndexer<String>(); langIndexer.index(new String[] {"spanish", "latin"}); langIndexer.lock();
		String[] chars = new String[] {" ","-","a","b","c","d","e","f","k","n","o","s","\\'o"};
		Indexer<String> charIndexer = new HashMapIndexer<String>(); charIndexer.index(chars);

		List<Integer> charIndices = new ArrayList<Integer>(); 
		for (String c : chars) charIndices.add(charIndexer.getIndex(c)); 
		Set<Integer> fullCharSet = makeSet(charIndices);
		@SuppressWarnings("unchecked")
		Set<Integer>[] activeCharacterSets = new Set[] {fullCharSet, fullCharSet};
		charIndexer.getIndex("z");
		charIndexer.getIndex(Charset.LONG_S);
		for (String c : new String[] {"a","b","c","d","e","f","k","n","o","s","z"}) charIndices.add(charIndexer.getIndex(c+TILDE_COMBINING));
		charIndexer.lock();
		double gsmPower = 2.0; 
		int minCountsForEvalGsm = 2;
		String outputPath = ""; 
		
		BasicGlyphSubstitutionModelFactory gsmf = new BasicGlyphSubstitutionModelFactory(
				gsmSmoothingCount,
				gsmElisionSmoothingCountMultiplier,
				langIndexer, 
				charIndexer, 
				activeCharacterSets,
				gsmPower, 
				minCountsForEvalGsm,
				outputPath);

		assertEquals(gsmSmoothingCount*gsmElisionSmoothingCountMultiplier, gsmf.getSmoothingValue(0, charIndexer.getIndex("\\'o"), gsmf.GLYPH_ELISION_TILDE), 1e-9);
		assertEquals(gsmSmoothingCount, gsmf.getSmoothingValue(0, charIndexer.getIndex("k"), charIndexer.getIndex("k")), 1e-9);
		assertEquals(gsmSmoothingCount*gsmElisionSmoothingCountMultiplier, gsmf.getSmoothingValue(0, charIndexer.getIndex("k"), gsmf.GLYPH_FIRST_ELIDED), 1e-9);
		assertEquals(gsmSmoothingCount*gsmElisionSmoothingCountMultiplier, gsmf.getSmoothingValue(0, charIndexer.getIndex("k"), gsmf.GLYPH_FIRST_ELIDED), 1e-9);
		assertEquals(gsmSmoothingCount*gsmElisionSmoothingCountMultiplier, gsmf.getSmoothingValue(0, charIndexer.getIndex("k"), gsmf.GLYPH_TILDE_ELIDED), 1e-9);
		assertEquals(gsmSmoothingCount, gsmf.getSmoothingValue(0, charIndexer.getIndex("a"), charIndexer.getIndex("a")), 1e-9);
		assertEquals(gsmSmoothingCount*gsmElisionSmoothingCountMultiplier, gsmf.getSmoothingValue(0, charIndexer.getIndex("n"), gsmf.GLYPH_TILDE_ELIDED), 1e-9);
		assertEquals(gsmSmoothingCount, gsmf.getSmoothingValue(0, charIndexer.getIndex("a"), charIndexer.getIndex("a")), 1e-9);
		assertEquals(0.0, gsmf.getSmoothingValue(0, charIndexer.getIndex("a"), charIndexer.getIndex("z")), 1e-9);
		assertEquals(0.0, gsmf.getSmoothingValue(0, charIndexer.getIndex("a"), charIndexer.getIndex(Charset.LONG_S)), 1e-9);
		assertEquals(gsmSmoothingCount, gsmf.getSmoothingValue(0, charIndexer.getIndex("s"), charIndexer.getIndex(Charset.LONG_S)), 1e-9);

	}
	
}
