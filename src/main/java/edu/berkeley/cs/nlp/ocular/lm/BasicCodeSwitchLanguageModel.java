package edu.berkeley.cs.nlp.ocular.lm;

import indexer.Indexer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.model.LanguageTransitionPriors;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;

/**
 * TODO: Move some of the probability calculations from CodeSwitchTransitionModel to here?
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class BasicCodeSwitchLanguageModel implements CodeSwitchLanguageModel {
	private static final long serialVersionUID = 498752498346537382L;

	private Set<String> languages;

	/**
	 * Map[destinationLanguage -> destinationLM]
	 */
	private Map<String, SingleLanguageModel> subModels;

	/**
	 * Map[destinationLanguage -> destinationLM]
	 */
	private Map<String, Set<String>> wordLists;

	/**
	 * Map[destinationLanguage -> "prior prob of seeing destinationLanguage"]
	 */
	private Map<String, Double> languagePriors;

	/**
	 * Map[destinationLanguage -> Map[fromLanguage, "prior prob of switching from "from" to "destination"]]
	 */
	private Map<String, Map<String, Double>> languageTransitionPriors;

	private Indexer<String> charIndexer;
	private int maxOrder;

	public Set<String> languages() {
		return languages;
	}

	public SingleLanguageModel get(String language) {
		return subModels.get(language);
	}

	public Double languagePrior(String language) {
		return languagePriors.get(language);
	}

	public Double languageTransitionPrior(String fromLanguage, String destinationLanguage) {
		return languageTransitionPriors.get(destinationLanguage).get(fromLanguage);
	}

	public Set<String> wordList(String language) {
		return wordLists.get(language);
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	public int getMaxOrder() {
		return maxOrder;
	}

	public BasicCodeSwitchLanguageModel(Map<String, Tuple3<SingleLanguageModel, Set<String>, Double>> subModelsAndPriors, Indexer<String> charIndexer, double pKeepSameLanguage, int maxOrder) {
		if (subModelsAndPriors.isEmpty()) throw new IllegalArgumentException("languageModelsAndPriors may not be empty");
		if (pKeepSameLanguage <= 0.0 || pKeepSameLanguage >= 1.0) throw new IllegalArgumentException("pKeepSameLanguage on must be between 0 and 1 (it's " + pKeepSameLanguage + ")");

		// Total prob, for normalizing
		double languagePriorSum = 0.0;
		for (Map.Entry<String, Tuple3<SingleLanguageModel, Set<String>, Double>> lmAndPrior : subModelsAndPriors.entrySet()) {
			double prior = lmAndPrior.getValue()._3;
			if (prior <= 0.0) throw new IllegalArgumentException("prior on " + lmAndPrior.getKey() + " is not positive (it's " + prior + ")");
			languagePriorSum += prior;
		}

		this.languages = new HashSet<String>();
		this.languages.addAll(subModelsAndPriors.keySet());
		
		this.subModels = new HashMap<String, SingleLanguageModel>();
		this.languagePriors = new HashMap<String, Double>();
		this.wordLists = new HashMap<String, Set<String>>();
		for (Map.Entry<String, Tuple3<SingleLanguageModel, Set<String>, Double>> lmAndPrior : subModelsAndPriors.entrySet()) {
			String language = lmAndPrior.getKey();
			this.subModels.put(language, lmAndPrior.getValue()._1);
			this.languagePriors.put(language, lmAndPrior.getValue()._3 / languagePriorSum);
			this.wordLists.put(language, lmAndPrior.getValue()._2);
		}
		

		this.languageTransitionPriors = LanguageTransitionPriors.makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);

		this.charIndexer = charIndexer;
		this.maxOrder = maxOrder;
	}

	/**
	 * TODO: This is really just here for DenseBigramTransitionModel to use.
	 *       I have *NO IDEA* whether it matters that this doesn't consider: 
	 *         a) the role of spaces in determining language switches (it 
	 *            kind of assumes that every character can be a different 
	 *            language)
	 *         b) languageTransitionPriors (since we can't track the language
	 *            of the context)
	 */
	public double getCharNgramProb(int[] context, int c) {
		//			if(context[context.length-1] == charIndexer.getIndex(Main.SPACE)) { // this is right after a space  
		// assume any language is possible
		double probSum = 0.0;
		for (String language : languages) {
			probSum += subModels.get(language).getCharNgramProb(context, c) * languagePriors.get(language);
		}
		return probSum;
		//			}
		//			else{
		//				
		//			}
	}
}
