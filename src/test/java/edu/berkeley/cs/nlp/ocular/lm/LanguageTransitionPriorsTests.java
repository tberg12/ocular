package edu.berkeley.cs.nlp.ocular.lm;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeMap;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;

import java.util.Map;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class LanguageTransitionPriorsTests {

	@Test
	public void test_makeLanguageTransitionPriors_multipleLanguages() {

		Map<String, Double> languagePriors = makeMap( //
				makeTuple2("spanish", 0.5), //
				makeTuple2("latin", 0.3), //
				makeTuple2("nahautl", 0.1));
		double pKeepSameLanguage = 0.8;

		// Map[destinationLanguage -> (destinationLM, Map[fromLanguage, "prob of switching from "from" to "to"])]
		Map<String, Map<String, Double>> languageTransitionPriors = BasicCodeSwitchLanguageModel.makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);

		assertEquals((0.1 * 0.5) / 0.16, languageTransitionPriors.get("spanish").get("nahautl"), 1e-9);
		assertEquals((0.1 * 0.3) / 0.16, languageTransitionPriors.get("latin").get("nahautl"), 1e-9);
		assertEquals((0.8 * 0.1) / 0.16, languageTransitionPriors.get("nahautl").get("nahautl"), 1e-9);

		assertEquals((0.1 * 0.5) / 0.30, languageTransitionPriors.get("spanish").get("latin"), 1e-9);
		assertEquals((0.8 * 0.3) / 0.30, languageTransitionPriors.get("latin").get("latin"), 1e-9);
		assertEquals((0.1 * 0.1) / 0.30, languageTransitionPriors.get("nahautl").get("latin"), 1e-9);

		assertEquals((0.8 * 0.5) / 0.44, languageTransitionPriors.get("spanish").get("spanish"), 1e-9);
		assertEquals((0.1 * 0.3) / 0.44, languageTransitionPriors.get("latin").get("spanish"), 1e-9);
		assertEquals((0.1 * 0.1) / 0.44, languageTransitionPriors.get("nahautl").get("spanish"), 1e-9);
	}

	@Test
	public void test_makeLanguageTransitionPriors_oneLanguage() {
		Map<String, Double> languagePriors = makeMap( //
		makeTuple2("spanish", 0.5));
		double pKeepSameLanguage = 0.8;

		// Map[destinationLanguage -> (destinationLM, Map[fromLanguage, "prob of switching from "from" to "to"])]
		Map<String, Map<String, Double>> languageTransitionPriors = BasicCodeSwitchLanguageModel.makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);

		assertEquals(1.0, languageTransitionPriors.get("spanish").get("spanish"), 1e-9);
	}

	@Test
	public void test_makeLanguageTransitionPriors_noLanguages() {
		Map<String, Double> languagePriors = makeMap();
		double pKeepSameLanguage = 0.8;

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("languagePriors may not be empty", e.getMessage());
		}
	}

	@Test
	public void test_makeLanguageTransitionPriors_pKeepSameLanguageGreaterThan1() {
		Map<String, Double> languagePriors = makeMap( //
				makeTuple2("spanish", 0.5), //
				makeTuple2("latin", 0.3), //
				makeTuple2("nahautl", 0.1));
		double pKeepSameLanguage = 1.1;

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("pKeepSameLanguage must be between 0 and 1, was 1.1", e.getMessage());
		}
	}

	@Test
	public void test_makeLanguageTransitionPriors_pKeepSameLanguageZer0() {
		Map<String, Double> languagePriors = makeMap( //
				makeTuple2("spanish", 0.5), //
				makeTuple2("latin", 0.3), //
				makeTuple2("nahautl", 0.1));
		double pKeepSameLanguage = 0.0;

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("pKeepSameLanguage must be between 0 and 1, was 0.0", e.getMessage());
		}
	}

	@Test
	public void test_makeLanguageTransitionPriors_languagePriorZero() {
		Map<String, Double> languagePriors = makeMap( //
				makeTuple2("spanish", 0.5), //
				makeTuple2("latin", 0.0), //
				makeTuple2("nahautl", 0.2));
		double pKeepSameLanguage = 0.8;

		try {
			BasicCodeSwitchLanguageModel.makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);
			fail("exception expected");
		}
		catch (RuntimeException e) {
			assertEquals("prior on latin is not positive (it's 0.0)", e.getMessage());
		}
	}
}
