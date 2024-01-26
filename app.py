"""1-misol

# class Solution:
#     def defangIPaddr(self, address: str) -> str:
#         return address.replace(".", "[.]")
# # """

# 2-misol
"""class Solution:
    def buildArray(self, nums: list[int]) -> list[int]:
        new_list = []
        for i in nums:
            new_list.append(nums[i])

        return new_list
"""

# 3 mispl
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

# 4-misol: 35. Search Insert Position

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

# misol-7: 2769. Find the Maximum Achievable Number
"""class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num + (t * 2)"""

# misol-8: 1929. Concatenation of Array

"""class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        return nums+nums"""

# misol-9: 1512. Yaxshi juftliklar soni

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

# misol-10:1470. Shuffle the Array
"""
class Solution:
    def shuffle(self, nums: list[int], n: int) -> list[int]:
        new_list = []
        for i in range(n):
            new_list.append(nums[i])
            new_list.append(nums[i + n])
        return new_list"""

# misol-11: 2942. Find Words Containing Character
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

# misol-12: 2235. Add Two Integers
"""class Solution:
    def sum(self, num1: int, num2: int) -> int:
        return num1 + num2"""

# misol-13: 709. To Lower Case
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

# misol-15: 1859. Sorting the Sentence
"""class Solution:
    def sortSentence(self, s: str) -> str:
        # Split the shuffled sentence into a list of words
        words = s.split()

        # Sort the words based on the appended numbers
        sorted_words = sorted(words, key=lambda word: int(word[-1]))

        # Remove the numbers from each word
        original_sentence = ' '.join(word[:-1] for word in sorted_words)

        return original_sentence"""

# misol-16:1281. Subtract the Product and Sum of Digits of an Integer
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
# misol-17: 2114. Maximum Number of Words Found in Sentences
"""
class Solution:
    def mostWordsFound(self, sentences: list[str]) -> int:
        new_list = []
        for i in sentences:
            son = i.split()
            uzinlik = len(son)
            new_list.append(uzinlik)
        rezalt = max(new_list)
        return rezalt
"""

# 18-misol: 1. Two Sum
"""
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        num_dict = {}

        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_dict:
                return [num_dict[complement], i]
            num_dict[num] = i

        # No valid solution found
        return []
2 ushuli
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        length = len(nums)
        for i in range(length):
            for j in range(i + 1, length):
                if nums[i] + nums[j] == target:
                    return [i,j]
"""

# 19-misol: 12. Integer to Roman
"""
class Solution:
    def intToRoman(self, num: int) -> str:
        # Creating Dictionary for Lookup
        num_map = {
            1: "I",
            5: "V", 4: "IV",
            10: "X", 9: "IX",
            50: "L", 40: "XL",
            100: "C", 90: "XC",
            500: "D", 400: "CD",
            1000: "M", 900: "CM",
        }

        # Result Variable
        r = ''

        for n in [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]:
            # If n in list then add the roman value to result variable
            while n <= num:
                r += num_map[n]
                num -= n
        return r
"""

# 20-misol: 1704. Determine if String Halves Are Alike
# """
# class Solution:
#     def halvesAreAlike(self, s: str) -> bool:
#         def unli_son(strng):
#             """bu funksiya str qiymat qabul qiladi va unda nechta unli har qatnashganligini qaytaradi int:"""
#             unli = set('aeiouAEIOU')# unliy harflar
#             result = 0# bu yerda unli sonlarni sonini xisoblab boradi
#             for i in strng:# bu yerda kelgan str elimintlarini aylantiradi
#                 if i in unli:#elimentlarni tekshiradi unli
#                     result += 1# agar unliy bulsa result ga 1 qushib quyadi
#             return result# va qiymatni qaytaradi turi int
#         uzunlik = len(s)#bizga berilgan asil str ning uzunligini olamiz
#         yarimi = uzunlik // 2# va uni urtasini topish uchun uni 2 ga bulamiz
#         birinchi_qism = s[:yarimi]# bu yerda str ning boshidan yarmigacha bulgan qismini olamiz
#         ikkinchi_qism = s[yarimi:]# bu yerda yarmidan oxirigacha olamiz
#
# return unli_son(birinchi_qism) == unli_son(ikkinchi_qism)# va nixoyat funksiyadan qaytgan natijalarni solishtirib
# javab qaytaramiz """

# 21-misol: 1026. Maximum Difference Between Node and Ancestor

"""
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = self.right = None

class Solution:
    def maxAncestorDiff(self, root):
        m = [0]
        self.dfs(root, m)
        return m[0]

    def dfs(self, root, m):
        if not root:
            return float('inf'), float('-inf')

        left = self.dfs(root.left, m)
        right = self.dfs(root.right, m)

        min_val = min(root.val, min(left[0], right[0]))
        max_val = max(root.val, max(left[1], right[1]))

        m[0] = max(m[0], max(abs(min_val - root.val), abs(max_val - root.val)))

        return min_val, max_val

"""

# 22-misol: 2974. Minimum Number Game
"""
class Solution:
    def numberGame(self, nums: list[int]) -> list[int]:
        nums.sort()# listni tartiblab olamiz
        result = []
        for i in range(0, len(nums), 2):#bu yerda listni uzunligini oldim va uni range da 2 qadam bilan yurgizib oldim
            result.append(nums[i + 1])#Bob olgan songa teng buladi chunki u Alice dan kiyin oladi
            result.append(nums[i])#bu Alice olgani

        return result
"""
#23-misol: 2225. Find Players With Zero or One Losses
"""
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        match_map = {}
        
        for el in matches:
            won = el[0]
            lost = el[1]
            if won not in match_map:
               match_map[won] = [0,0]
            
            if lost not in match_map:
                match_map[lost] = [0,0]
            
            match_map[won][0] += 1
            match_map[lost][1] -= 1
        
        ans = [[],[]]
        for key in sorted(match_map):
            won = match_map[key][0]
            lost = match_map[key][1]
            if lost == 0:
                ans[0].append(key)
            elif lost == -1:
                ans[1].append(key)

        return ans
        
"""


#misol-24: 21. Merge Two Sorted Lists
"""class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy

        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next

            current = current.next

        # If one of the lists is not empty, append the remaining nodes
        if list1:
            current.next = list1
        elif list2:
            current.next = list2

        return dummy.next

"""

#misol : 1207. Unique Number of Occurrences

"""

class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        return len(set(Counter(arr).values()))==len(Counter(arr).values())
"""


# misol: 907. Sum of Subarray Minimums

"""
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        left = [-1] * n 
        right = [n] * n
        stack = []
        for i, value in enumerate(arr):
            while stack and arr[stack[-1]] >= value:  
                stack.pop()  
            if stack:
                left[i] = stack[-1]  
            stack.append(i) 

        stack = [] 
        for i in range(n - 1, -1, -1):  
            while stack and arr[stack[-1]] > arr[i]: 
                stack.pop()  
            if stack:
                right[i] = stack[-1]  
            stack.append(i) 

        mod = 10**9 + 7 
        result = sum((i - left[i]) * (right[i] - i) * value for i, value in enumerate(arr)) % mod
      
        return result 
"""


# misol: 9. Palindrome Number
"""class Solution:
    def isPalindrome(self, x: int) -> bool:
        son = str(x)
        return son == son[::-1]
"""

"""
#misol 2974. Minimum Number Game
class Solution:
    def numberGame(self, nums: list[int]) -> list[int]:
        nums.sort() # bu yerdakelgan listni tartiblab olamiz
        arr = [] # bu bush tuplam
        while nums:# numsda elimint bulsa true qiymat qaytaradi aks xolda false qiymat qaytarib takrorklanish tuxtaydi
            alic_move = nums.pop(0) #Alic  sonini olishi
            bob_move = nums.pop(0) #bob sonini olishi
            arr.append(bob_move)
            arr.append(alic_move)
        return arr
"""

#misol 576. Out of Boundary Paths

"""
class Solution:
    MOD = 10**9 + 7

    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        # Memoization dictionary to store the number of paths for each cell at each move
        memo = {}
        
        def dfs(x, y, move):
            # If the move count exceeds maxMove or the cell is out of bounds, return 0
            if move > maxMove or x < 0 or x >= m or y < 0 or y >= n:
                return 1
            
            # If the number of moves left is zero, there are no more valid moves
            if move == 0:
                return 0
            
            # If the result for the current cell at the current move is already calculated, return it
            if (x, y, move) in memo:
                return memo[(x, y, move)]
            
            # Calculate the number of paths recursively by considering all four possible moves
            paths = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                paths += dfs(x + dx, y + dy, move - 1)
                paths %= self.MOD
            
            # Memoize the result for the current cell at the current move
            memo[(x, y, move)] = paths
            return paths
        
        # Start the depth-first search from the starting cell with maxMove moves left
        return dfs(startRow, startColumn, maxMove)

# Example usage:
m = 2
n = 2
maxMove = 2
startRow = 0
startColumn = 0
solution = Solution()
print(solution.findPaths(m, n, maxMove, startRow, startColumn))  # Output: 6

"""