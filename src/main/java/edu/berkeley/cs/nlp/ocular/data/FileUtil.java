package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;

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
		String[] split = name.split("\\.");
		return split[split.length - 1];
	}

	public static String withoutExtension(String name) {
		return name.replaceAll("\\.[^.]*$", "");
	}


	public static String pathRelativeTo(String fn1, String dir) {
		try {
			String selfPath = new File(fn1).getCanonicalPath();
			String dirPath = new File(dir).getCanonicalPath(); 
			
			List<String> as = pathToNameList(new File(selfPath));
			List<String> bs = pathToNameList(new File(dirPath));
			
			int longestCommonPrefix = 0;
			int aLen = as.size();
			int bLen = bs.size();
			while (longestCommonPrefix < aLen && longestCommonPrefix < bLen && as.get(longestCommonPrefix).equals(bs.get(longestCommonPrefix))) 
				++longestCommonPrefix;
			
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
