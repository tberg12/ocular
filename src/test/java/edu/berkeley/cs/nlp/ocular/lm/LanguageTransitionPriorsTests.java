package edu.berkeley.cs.nlp.ocular.lm;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.List;

import org.junit.Test;

import tberg.murphy.indexer.HashMapIndexer;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class LanguageTransitionPriorsTests {

	@Test
	public void test_makeLanguageTransitionProbs_multipleLanguages() {

		List<Double> languagePriors = makeList( //
				0.5, // Tuple2("spanish", 0.5), //
				0.3, // Tuple2("latin", 0.3), //
				0.1); //Tuple2("nahautl", 0.1));
		double pKeepSameLanguage = 0.8;
		Indexer<String> langIndexer = new HashMapIndexer<String>();
		langIndexer.index(new String[] { "spanish", "latin", "nahuatl" } );

		// Map[destinationLanguage -> (destinationLM, Map[fromLanguage, "prob of switching from "from" to "to"])]
		List<List<Double>> languageTransitionPriors = BasicCodeSwitchLanguageModel.makeLanguageTransitionProbs(languagePriors, pKeepSameLanguage, langIndexer);

		assertEquals((0.1 * 0.5) / 0.16, languageTransitionPriors.get(langIndexer.getIndex("spanish")).get(langIndexer.getIndex("nahuatl")), 1e-9);
		assertEquals((0.1 * 0.3) / 0.16, languageTransitionPriors.get(langIndexer.getIndex("latin")).get(langIndexer.getIndex("nahuatl")), 1e-9);
		assertEquals((0.8 * 0.1) / 0.16, languageTransitionPriors.get(langIndexer.getIndex("nahuatl")).get(langIndexer.getIndex("nahuatl")), 1e-9);

		assertEquals((0.1 * 0.5) / 0.30, languageTransitionPriors.get(langIndexer.getIndex("spanish")).get(langIndexer.getIndex("latin")), 1e-9);
		assertEquals((0.8 * 0.3) / 0.30, languageTransitionPriors.get(langIndexer.getIndex("latin")).get(langIndexer.getIndex("latin")), 1e-9);
		assertEquals((0.1 * 0.1) / 0.30, languageTransitionPriors.get(langIndexer.getIndex("nahuatl")).get(langIndexer.getIndex("latin")), 1e-9);

		assertEquals((0.8 * 0.5) / 0.44, languageTransitionPriors.get(langIndexer.getIndex("spanish")).get(langIndexer.getIndex("spanish")), 1e-9);
		assertEquals((0.1 * 0.3) / 0.44, languageTransitionPriors.get(langIndexer.getIndex("latin")).get(langIndexer.getIndex("spanish")), 1e-9);
		assertEquals((0.1 * 0.1) / 0.44, languageTransitionPriors.get(langIndexer.getIndex("nahuatl")).get(langIndexer.getIndex("spanish")), 1e-9);
	}

	@Test
	public void test_makeLanguageTransitionProbs_oneLanguage() {
		List<Double> languagePriors = makeList( //
				0.5); //Tuple2("spanish", 0.5));
		double pKeepSameLanguage = 0.8;
		Indexer<String> langIndexer = new HashMapIndexer<String>();
		langIndexer.index(new String[] { "spanish" } );

		// Map[destinationLanguage -> (destinationLM, Map[fromLanguage, "prob of switching from "from" to "to"])]
		List<List<Double>> languageTransitionPriors = BasicCodeSwitchLanguageModel.makeLanguageTransitionProbs(languagePriors, pKeepSameLanguage, langIndexer);

		assertEquals(1.0, languageTransitionPriors.get(langIndexer.getIndex("spanish")).get(langIndexer.getIndex("spanish")), 1e-9);
	}

	@Test
	public void test_makeLanguageTransitionProbs_noLanguages() {
		List<Double> languagePriors = makeList();
		double pKeepSameLanguage = 0.8;
		Indexer<String> langIndexer = new HashMapIndexer<String>();

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionProbs(languagePriors, pKeepSameLanguage, langIndexer);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("languagePriors may not be empty", e.getMessage());
		}
	}

	@Test
	public void test_makeLanguageTransitionProbs_pKeepSameLanguageGreaterThan1() {
		List<Double> languagePriors = makeList( //
				0.5, // Tuple2("spanish", 0.5), //
				0.3, // Tuple2("latin", 0.3), //
				0.1); //Tuple2("nahautl", 0.1));
		double pKeepSameLanguage = 1.1;
		Indexer<String> langIndexer = new HashMapIndexer<String>();
		langIndexer.index(new String[] { "spanish", "latin", "nahuatl" } );

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionProbs(languagePriors, pKeepSameLanguage, langIndexer);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("pKeepSameLanguage must be between 0 and 1, was 1.1", e.getMessage());
		}
	}

	@Test
	public void test_makeLanguageTransitionProbs_pKeepSameLanguageZero() {
		List<Double> languagePriors = makeList( //
				0.5, // Tuple2("spanish", 0.5), //
				0.3, // Tuple2("latin", 0.3), //
				0.1); //Tuple2("nahautl", 0.1));
		double pKeepSameLanguage = 0.0;
		Indexer<String> langIndexer = new HashMapIndexer<String>();
		langIndexer.index(new String[] { "spanish", "latin", "nahuatl" } );

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionProbs(languagePriors, pKeepSameLanguage, langIndexer);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("pKeepSameLanguage must be between 0 and 1, was 0.0", e.getMessage());
		}
	}

	@Test
	public void test_makeLanguageTransitionProbs_languagePriorZero() {
		List<Double> languagePriors = makeList( //
				0.5, // Tuple2("spanish", 0.5), //
				0.0, // Tuple2("latin", 0.0), //
				0.2); //Tuple2("nahautl", 0.2));
		double pKeepSameLanguage = 0.8;
		Indexer<String> langIndexer = new HashMapIndexer<String>();
		langIndexer.index(new String[] { "spanish", "latin", "nahuatl" } );

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionProbs(languagePriors, pKeepSameLanguage, langIndexer);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("prior on latin is not positive (it's 0.0)", e.getMessage());
		}
	}
}
