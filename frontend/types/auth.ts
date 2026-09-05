export interface User {
  id: string;
  email: string;
  name: string;
  full_name?: string;
  profile?: Record<string, any>;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at?: string;
}

export interface UserLogin {
  email: string;
  password: string;
}

export interface UserRegister {
  email: string;
  password: string;
  name?: string;
  full_name?: string;
  profile?: Record<string, any>;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}
