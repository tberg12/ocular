package edu.berkeley.cs.nlp.ocular.util;

import java.util.List;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
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

	public static String drop(String s, int n) {
		if (n <= 0)
			return s;
		else if (n < s.length())
			return s.substring(n);
		else
			return "";
	}

	public static String last(String s) {
		if (s.isEmpty()) throw new IllegalArgumentException("cannot get `last` of empty string");
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

	public static String join(List<String> xs, String sep) {
		int sepLen = sep.length();
		StringBuilder sb = new StringBuilder();
		for (String x : xs)
			sb.append(x).append(sep);
		return sb.length() > 0 ? sb.delete(sb.length() - sepLen, sb.length()).toString() : "";
	}

	public static boolean equals(String a, String b) {
		if (a == null)
			return b == null;
		else
			return a.equals(b);
	}
	
	public static int longestCommonPrefix(String a, String b) {
		int i = 0;
		char[] as = a.toCharArray();
		char[] bs = b.toCharArray();
		int aLen = as.length;
		int bLen = bs.length;
		while (i < aLen && i < bLen && as[i] == bs[i])
			++i;
		return i;
	}
	
}
