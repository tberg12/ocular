package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class FileUtil {

	public static List<File> recursiveFiles(String name) {
		return FileUtil.recursiveFiles(name, null);
	}

	public static List<File> recursiveFiles(File file) {
		return FileUtil.recursiveFiles(file, null);
	}

	public static List<File> recursiveFiles(String name, Set<String> validExtensions) {
		return FileUtil.recursiveFiles(new File(name), validExtensions);
	}

	public static List<File> recursiveFiles(File file, Set<String> validExtensions) {
		List<File> files = new ArrayList<File>();
		if (file.isDirectory()) {
			for (File f : file.listFiles()) {
				if (!f.getName().startsWith(".")) { // ignore hidden files
					files.addAll(recursiveFiles(f, validExtensions));
				}
			}
		}
		else {
			if (validExtensions == null || validExtensions.contains(extension(file.getName()))) {
				files.add(file);
			}
		}
		return files;
	}

	public static List<File> recursiveFiles(List<String> names) {
		return FileUtil.recursiveFiles(names, null);
	}

	public static List<File> recursiveFiles(List<String> names, Set<String> validExtensions) {
		List<File> files = new ArrayList<File>();
		for (String f : names)
			files.addAll(FileUtil.recursiveFiles(f, validExtensions));
		return files;
	}
	
	public static String extension(String name) {
		int dotIdx = name.lastIndexOf(".");
		return dotIdx >= 0 ? name.substring(dotIdx + 1) : null;
	}

	public static String withoutExtension(String name) {
		int dotIdx = name.lastIndexOf(".");
		return dotIdx >= 0 ? name.substring(0, dotIdx) : name;
	}


	/**
	 * @param fn1
	 * @param fn2
	 * @return
	 */
	public static Tuple2<String, String> removeCommonPathPrefix(File fn1, File fn2) {
		try {
			List<String> as = pathToNameList(fn1.getCanonicalFile());
			List<String> bs = pathToNameList(fn2.getCanonicalFile());
			
			int longestCommonPrefix = CollectionHelper.longestCommonPrefix(as, bs);
			
			String aSuffix = StringHelper.join(as.subList(longestCommonPrefix, as.size()), File.separator);
			String bSuffix = StringHelper.join(bs.subList(longestCommonPrefix, bs.size()), File.separator);
			return Tuple2.makeTuple2(aSuffix, bSuffix);
		}
		catch (IOException e) { throw new RuntimeException(e); }
	}
	
	/**
	 * @param fn1
	 * @param fn2
	 * @return
	 */
	public static Tuple2<String, String> removeCommonPathPrefixOfParents(File fn1, File fn2) {
		try {
			return removeCommonPathPrefix(fn1.getCanonicalFile().getParentFile(), fn2.getCanonicalFile().getParentFile());
		}
		catch (IOException e) { throw new RuntimeException(e); }
	}
	
	/**
	 * This will produce a result such that dir/result is the same file as fn1.
	 * 
	 * @param fn1
	 * @param dir
	 * @return
	 */
	public static String pathRelativeTo(String fn1, String dir) {
		try {
			List<String> as = pathToNameList(new File(new File(fn1).getCanonicalPath()));
			List<String> bs = pathToNameList(new File(new File(dir).getCanonicalPath()));

			int longestCommonPrefix = CollectionHelper.longestCommonPrefix(as, bs);

			List<String> prefix = CollectionHelper.fillList(bs.size()-longestCommonPrefix, "..");
			List<String> suffix = as.subList(longestCommonPrefix, as.size());
			return StringHelper.join(CollectionHelper.listCat(prefix, suffix), File.separator);
		} 
		catch (IOException e) { throw new RuntimeException(e); }
	}
	
	public static List<String> pathToNameList(File f) {
		List<String> l = new ArrayList<String>();
		while (f != null) {
			l.add(0, f.getName());
			f = f.getParentFile();
		}
		return l;
	}
	
	
}
