package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import tberg.murphy.fileio.f;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ReplaceSomeTextReader implements TextReader {

	private final List<Tuple2<Tuple2<List<String>, List<String>>, Integer>> rules;
	private final TextReader delegate;
	private final int[] occurrences;

	/**
	 * @param delegate
	 * @param rules	<<input, output>, each> Replace `input` by `output` every `each` occurrences 
	 */
	public ReplaceSomeTextReader(List<Tuple2<Tuple2<List<String>, List<String>>, Integer>> rules, TextReader delegate) {
		this.rules = rules;
		this.delegate = delegate;
		this.occurrences = new int[rules.size()];
	}

	public List<String> readCharacters(String line) {
		List<String> result = delegate.readCharacters(line);
		for (int i = 0; i < rules.size(); ++i) {
			Tuple2<Tuple2<List<String>, List<String>>, Integer> r = rules.get(i);
			List<String> input = r._1._1;
			List<String> output = r._1._2;
			int each = r._2;
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

	public static List<Tuple2<Tuple2<List<String>, List<String>>, Integer>> loadRulesFromFile(String path) {
		List<Tuple2<Tuple2<List<String>, List<String>>, Integer>> result = new ArrayList<Tuple2<Tuple2<List<String>, List<String>>, Integer>>();
		for (String line : f.readLines(path)) {
			if (!line.trim().isEmpty()) {
				String[] parts = line.split("\t");
				if (parts.length != 3) throw new RuntimeException("line does not contain 3 parts.  found: " + Arrays.asList(parts));
				result.add(Tuple2(Tuple2(Charset.readNormalizeCharacters(parts[0]), Charset.readNormalizeCharacters(parts[1])), Integer.valueOf(parts[2])));
			}
		}
		return result;
	}

	public String toString() {
		return "ReplaceSomeTextReader(rules=..., " + delegate + ")";
	}

}
