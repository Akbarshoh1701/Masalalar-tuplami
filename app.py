
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

# class Solution:
#     def searchInsert(self, nums: list[int], target: int) -> int:
#         if num == tar



#misol 5 == 13. Rimdan butun songa

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






