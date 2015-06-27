package edu.berkeley.cs.nlp.ocular.util;

import java.util.List;

public class StringHelper {

	public static String toUnicode(String s) {
		//if (s.length() != 1) throw new RuntimeException("toUnicode input must be a single character");
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < s.length(); ++i)
			sb.append(toUnicode(s.charAt(i)));
		return sb.toString();
	}

	public static String toUnicode(char c) {
		return "\\u" + Integer.toHexString(c | 0x10000).substring(1);
	}

	public static String take(String s, int n) {
		if (n <= 0)
			return "";
		else if (n < s.length())
			return s.substring(0, n);
		else
			return s;
	}

	public static String last(String s) {
		if (s.length() == 0) throw new IllegalArgumentException("cannot get `last` of empty string");
		return s.substring(s.length() - 1);
	}

	public static String join(String... xs) {
		StringBuilder sb = new StringBuilder();
		for (String x : xs)
			sb.append(x);
		return sb.toString();
	}

	public static String join(List<String> xs) {
		StringBuilder sb = new StringBuilder();
		for (String x : xs)
			sb.append(x);
		return sb.toString();
	}

}
