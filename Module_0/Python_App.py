from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI(title="My First Python App")


# Request/Response Model
class User(BaseModel):
    userId: int
    firstName: str
    lastName: str
    email: str
    phone: int
    isActive: bool

# In-memory storage
users = []

# Create User
@app.post("/users")
def create_user(user: User):
    users.append(user)
    return users


# Get All Users
# Get an item

@app.get("/users")
def get_users(): 
    userList = [user for user in users if user.isActive == 1]   
    return userList

# Get User By ID
@app.get("/users/{user_id}")
def get_user(user_id: int):
    userItem = next((u for u in users if u.userId == user_id and u.isActive == 1), None)
    return userItem


# Update User
@app.put("/users")
def update_user(updateUser: User):
    for userItem in users:
        if userItem.userId == updateUser.userId:
            userItem.firstName = updateUser.firstName
            userItem.lastName = updateUser.lastName
            userItem.email = updateUser.email
            userItem.phone = updateUser.phone
            return user
    return None


# Delete User
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    for index, user in enumerate(users):
        if user.userId == user_id:
            del users[index]
            return {"message": "User deleted"}

    raise HTTPException(status_code=404, detail="User not found")