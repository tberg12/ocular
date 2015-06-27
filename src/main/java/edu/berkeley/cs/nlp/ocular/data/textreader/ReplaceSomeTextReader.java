package edu.berkeley.cs.nlp.ocular.data.textreader;

import static tuple.Pair.makePair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import tuple.Pair;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import fileio.f;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class ReplaceSomeTextReader implements TextReader {

	private final List<Pair<Pair<List<String>, List<String>>, Integer>> rules;
	private final TextReader delegate;
	private final int[] occurrences;

	/**
	 * @param delegate
	 * @param rules	<<input, output>, each> Replace `input` by `output` every `each` occurrences 
	 */
	public ReplaceSomeTextReader(List<Pair<Pair<List<String>, List<String>>, Integer>> rules, TextReader delegate) {
		this.rules = rules;
		this.delegate = delegate;
		this.occurrences = new int[rules.size()];
	}

	public List<String> readCharacters(String line) {
		List<String> result = delegate.readCharacters(line);
		for (int i = 0; i < rules.size(); ++i) {
			Pair<Pair<List<String>, List<String>>, Integer> r = rules.get(i);
			List<String> input = r.getFirst().getFirst();
			List<String> output = r.getFirst().getSecond();
			int each = r.getSecond();
			List<String> newResult = new ArrayList<String>();
			for (int j = 0; j < input.size() - 1; ++j) {
				// add some buffer to the end so sliding goes to the end
				result.add(null);
			}
			Iterator<List<String>> iter = CollectionHelper.sliding(result, input.size());
			while (iter.hasNext()) {
				List<String> x = iter.next();
				if (x.equals(input)) {
					if (x.equals(input) && occurrences[i] % each == each - 1) {
						newResult.addAll(output); // add `output` to the result (to replace `input`)
						for (int j = 0; j < input.size() - 1; ++j) {
							//remove the rest of `input` from `iter`
							iter.next();
						}
					}
					else {
						newResult.add(x.get(0));
					}
					++occurrences[i];
				}
				else {
					newResult.add(x.get(0));
				}
			}
			result = newResult;
		}
		return result;
	}

	public static List<Pair<Pair<List<String>, List<String>>, Integer>> loadRulesFromFile(String path) {
		TextReader tr = new BasicTextReader();
		List<Pair<Pair<List<String>, List<String>>, Integer>> result = new ArrayList<Pair<Pair<List<String>, List<String>>, Integer>>();
		for (String line : f.readLines(path)) {
			if (!line.trim().isEmpty()) {
				String[] parts = line.split("\t");
				if (parts.length != 3) throw new RuntimeException("line does not contain 3 parts.  found: " + Arrays.asList(parts));
				result.add(makePair(makePair(tr.readCharacters(parts[0]), tr.readCharacters(parts[1])), Integer.valueOf(parts[2])));
			}
		}
		return result;
	}

	public String toString() {
		return "ReplaceSomeTextReader(rules=..., " + delegate + ")";
	}

}
