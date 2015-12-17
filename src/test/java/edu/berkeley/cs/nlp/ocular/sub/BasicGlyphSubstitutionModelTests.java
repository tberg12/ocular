package edu.berkeley.cs.nlp.ocular.sub;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.TILDE_ESCAPE;
import static edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType.*;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.sub.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import indexer.HashMapIndexer;
import indexer.Indexer;

public class BasicGlyphSubstitutionModelTests {

	@Test
	public void test_getSmoothingValue() {

		double gsmSmoothingCount = 0.1;
		double gsmElisionSmoothingCountMultiplier = 500.0;
		Indexer<String> langIndexer = new HashMapIndexer<String>(); langIndexer.index(new String[] {"spanish", "latin"}); langIndexer.lock();
		String[] chars = new String[] {" ","a","b","c","d","e","k","n","s"};
		Indexer<String> charIndexer = new HashMapIndexer<String>(); charIndexer.index(chars);

		List<Integer> charIndices = new ArrayList<Integer>(); 
		for (String c : chars) charIndices.add(charIndexer.getIndex(c)); 
		Set<Integer> fullCharSet = makeSet(charIndices);
		@SuppressWarnings("unchecked")
		Set<Integer>[] activeCharacterSets = new Set[] {fullCharSet, fullCharSet};
		for (String c : new String[] {"a","b","c","d","e","k","n","s"}) charIndices.add(charIndexer.getIndex(TILDE_ESCAPE+c)); 
		charIndexer.lock();
		double gsmPower = 2.0; 
		int minCountsForEvalGsm = 2;
		String inputPath = ""; 
		String outputPath = ""; 
		List<Document> documents = makeList(); 
		List<Document> evalDocuments = makeList();
		
		BasicGlyphSubstitutionModelFactory gsmf = new BasicGlyphSubstitutionModelFactory(
				gsmSmoothingCount,
				gsmElisionSmoothingCountMultiplier,
				langIndexer, 
				charIndexer, 
				activeCharacterSets, 
				gsmPower, 
				minCountsForEvalGsm, 
				inputPath, 
				outputPath, 
				documents, 
				evalDocuments);

		assertEquals(gsmSmoothingCount, gsmf.getSmoothingValue(0, ELISION_TILDE, charIndexer.getIndex("s"), charIndexer.getIndex("k"), gsmf.GLYPH_ELIDED), 1e-9);
		assertEquals(gsmSmoothingCount, gsmf.getSmoothingValue(0, ELIDED, charIndexer.getIndex(" "), charIndexer.getIndex("a"), charIndexer.getIndex("a")), 1e-9);
		assertEquals(gsmSmoothingCount*gsmElisionSmoothingCountMultiplier, gsmf.getSmoothingValue(0, ELISION_TILDE, charIndexer.getIndex("e"), charIndexer.getIndex("n"), gsmf.GLYPH_ELIDED), 1e-9);
		assertEquals(gsmSmoothingCount, gsmf.getSmoothingValue(0, NORMAL_CHAR, charIndexer.getIndex(" "), charIndexer.getIndex("a"), charIndexer.getIndex("a")), 1e-9);

	}
	
}
