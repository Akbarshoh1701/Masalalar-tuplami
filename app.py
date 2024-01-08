
"""1-misol

# class Solution:
#     def defangIPaddr(self, address: str) -> str:
#         return address.replace(".", "[.]")
# # """


#2-misol
"""class Solution:
    def buildArray(self, nums: list[int]) -> list[int]:
        new_list = []
        for i in nums:
            new_list.append(nums[i])

        return new_list
"""

#3 mispl
"""class Solution:
    def finalValueAfterOperations(self, operations: list[str]) -> int:
        X = 0
        for i in operations:
            if i == "--X" or i == "X--":
                X -= 1
            elif i == "X++" or "++X":
                X += 1

        return X

"""

#4-misol 35. Search Insert Position

"""class Solution:
    def searchInsert(self, nums: list[int], target: int) -> int:
        for i in range(len(nums)):
            if nums[i] >= target:
                return i
        return len(nums)

#clasni chaqirib olamiz
solution_instance = Solution()

# Misolda searchInsert usulini chaqiring
result = solution_instance.searchInsert([1, 2, 3, 5], 4)
print(result)"""


""" #misol 5 == 13. Rimdan butun songa

class Solution:
    def romanToInt(self, s: str) -> int:
        son = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        rezalt = 0
        prev_value = 0
        s = s.upper()

        for i in s:
            if i not in son:
                return "Noto'g'ri belgi kiritildi"

            current_value = son[i]
            if current_value > prev_value:
                rezalt += current_value - 2 * prev_value
            else:
                rezalt += current_value
            prev_value = current_value

        return rezalt


# Create an instance of the Solution class
solution_instance = Solution()

# Call the romanToInt method on the instance
result = solution_instance.romanToInt("Iv")

# Print the result
print(result)
"""

# misol-6, 446. Arithmetic Slices II - Subsequence

"""class Solution:
    def numberOfArithmeticSlices(self, nums: list[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0

        total_count = 0
        dp = [{} for _ in range(n)]

        for i in range(n):
            for j in range(i):
                diff = nums[i] - nums[j]
                dp[i][diff] = dp[i].get(diff, 0) + 1

                if diff in dp[j]:
                    dp[i][diff] += dp[j][diff]
                    total_count += dp[j][diff]

        return total_count

# Example usage:
# Create an instance of the Solution class
solution = Solution()

# Test the example
nums = [2, 4, 6, 8, 10]
output = solution.numberOfArithmeticSlices(nums)
print("Number of arithmetic subsequences:", output)
"""

#misol-7: 2769. Find the Maximum Achievable Number
"""class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num + (t * 2)"""



#misol-8: 1929. Concatenation of Array

"""class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        return nums+nums"""


#misol-9: 1512. Yaxshi juftliklar soni

"""class Solution:
    def numIdenticalPairs(self, nums: list[int]) -> int:
        count_dict = {}
        result = 0

        for num in nums:
            count_dict[num] = count_dict.get(num, 0) + 1

        for count in count_dict.values():
            if count >= 2:
                result += (count * (count - 1)) // 2

        return result

-------ikkinchu usuli--------
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        pairs = 0
        for i in range(len(nums)-1):
            for j in range((i+1), len(nums)):
                if nums[i] == nums[j]:
                    pairs += 1
        return pairs

"""


#misol-10:1470. Shuffle the Array
"""
class Solution:
    def shuffle(self, nums: list[int], n: int) -> list[int]:
        new_list = []
        for i in range(n):
            new_list.append(nums[i])
            new_list.append(nums[i + n])
        return new_list"""

#misol-11: 2942. Find Words Containing Character
"""
from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        new_list = []
        for i, word in enumerate(words):
            if x in word:
                new_list.append(i)
        return new_list

---ikkinchi usul qulayi----

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        return [i for i in range(len(words)) if x in words[i]]
"""

#misol-12: 2235. Add Two Integers
"""class Solution:
    def sum(self, num1: int, num2: int) -> int:
        return num1 + num2"""

#misol-13: 709. To Lower Case
"""
class Solution:
    def toLowerCase(self, s: str) -> str:
        return s.lower()
"""

# misol-14:1913. Maximum Product Difference Between Two Pairs
"""
class Solution:
    def maxProductDifference(self, nums: list[int]) -> int:
        nums.sort()# listni tartiblab olamiz
        max1, max2 = nums[-1], nums[-2]
        min1, min2 = nums[0], nums[1]
        return (max1 * max2) - (min1 * min2)

solution = Solution()
nums = [4, 2, 5, 9, 7, 4, 8]
result = solution.maxProductDifference(nums)
print(result)
"""

#misol-15: 1859. Sorting the Sentence
"""class Solution:
    def sortSentence(self, s: str) -> str:
        # Split the shuffled sentence into a list of words
        words = s.split()

        # Sort the words based on the appended numbers
        sorted_words = sorted(words, key=lambda word: int(word[-1]))

        # Remove the numbers from each word
        original_sentence = ' '.join(word[:-1] for word in sorted_words)

        return original_sentence"""

#misol-16:1281. Subtract the Product and Sum of Digits of an Integer
"""
class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        n = str(n)
        kupaytma = 1
        yigindi = 0
        for i in n:
            i = int(i)
            kupaytma *= i
            yigindi += i
        return kupaytma - yigindi
    """











